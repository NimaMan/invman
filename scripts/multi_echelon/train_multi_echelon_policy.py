"""Design + train a learned policy for a multi-echelon (Gijs 2022 / Van Roy 1997) instance.

Objective
---------
Run the policy-design-plus-CMA-ES method on any reference instance of the
divergent_special_delivery family and report, in one run:

1. Literature reproduction (only for instances that carry a published constant base-stock
   cost -- the van_roy_1997 reproduction instances): the cost at the published (yw, yr)
   levels vs the published number (e.g. simple 51.7, case_study1 1302, case_study2 1449).
   The paper-faithful gijs_2022 search targets carry no published row, so this is skipped
   for them (their absolute Van Roy cost reproduces on the sibling van_roy1997_* instance).
2. The in-env best constant base-stock (grid search) -- the benchmark the learned policy
   must beat.
3. A soft-tree depth sweep trained by CMA-ES, with the relative improvement over the best
   constant base-stock. For the faithful gijs_2022 settings that relative improvement is the
   comparator to Gijs's published A3C savings (setting 1: 8.95%, setting 2: 12.09%).

Algorithm
---------
build_reference_args(reference) pins the env parameters and discrete base-stock action grids
from the invman_rust catalog (single source of truth). The benchmark and the literature
reproduction are computed by multi_echelon_search_stationary_policy (Rust). For each tree
depth: a soft_tree (oblique splits, linear leaves) over the raw decision-state observation,
with the policy-owned divide-by-scale normalizer, is optimised by CMA-ES (run_experiment),
then evaluated at a longer horizon over several seeds. The best architecture's model and a
JSON report are persisted under outputs/multi_echelon/<reference>_policy/.

Usage
-----
    python scripts/multi_echelon/train_multi_echelon_policy.py --reference gijsbrechts2022_setting1 --budget full
    python scripts/multi_echelon/train_multi_echelon_policy.py --reference van_roy1997_simple_problem --budget full
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import importlib.util

import numpy as np

import invman_rust  # noqa: F401  (ensures the extension is importable before training)
from invman.experiment_runner import build_model, evaluate_model, run_experiment
from invman import rollout_fitness
from invman.es_mp import train as _es_train
from invman.cpu_limits import normalize_args_cpu_limits
from invman.policy_registry import apply_policy_name, get_policy_spec, make_soft_tree_policy_name
from invman.utils import set_global_seeds

# Reuse build_reference_args / best_constant_base_stock_baseline from the autoresearch entrypoint.
_AUTORESEARCH = PACKAGE_ROOT / "scripts" / "multi_echelon" / "autoresearch_multi_echelon.py"
_spec = importlib.util.spec_from_file_location("autoresearch_multi_echelon", _AUTORESEARCH)
_are = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_are)

BUDGETS = {
    "screening": {"training_episodes": 200, "es_population": 16, "horizon": 2000, "eval_horizon": 20000, "eval_seeds": 5},
    "full": {"training_episodes": 400, "es_population": 24, "horizon": 3000, "eval_horizon": 30000, "eval_seeds": 6},
}

# Faithful gijs_2022 settings carry no published rows; the published A3C comparator lives on
# the sibling van_roy_1997 reproduction instance.
_A3C_SIBLING = {
    "gijsbrechts2022_setting1": "van_roy1997_case_study1",
    "gijsbrechts2022_setting2": "van_roy1997_case_study2",
}


def published_a3c_savings(reference_name):
    reference = dict(invman_rust.multi_echelon_get_reference_instance(reference_name))
    if reference.get("published_a3c_savings_pct") is not None:
        return reference["published_a3c_savings_pct"]
    sibling = _A3C_SIBLING.get(reference_name)
    if sibling is not None:
        return dict(invman_rust.multi_echelon_get_reference_instance(sibling)).get("published_a3c_savings_pct")
    return None


def literature_reproduction(args, reference_name, horizon, replications, seed):
    """Cost at the published constant base-stock levels vs the published number, or None if
    the instance carries no published row (the faithful gijs_2022 search targets)."""
    reference = dict(invman_rust.multi_echelon_get_reference_instance(reference_name))
    published = reference.get("published_constant_base_stock_mean_cost")
    levels = [int(v) for v in reference.get("published_constant_base_stock_levels", [])]
    if published is None or len(levels) != 2:
        return None
    result = invman_rust.multi_echelon_search_stationary_policy(
        policy_kind="constant_base_stock",
        allocation_mode="min_shortage",
        warehouse_levels=[levels[0]],
        retailer_levels=[levels[1]],
        warehouse_lead_time=int(args.warehouse_lead_time),
        retailer_lead_time=int(args.retailer_lead_time),
        num_retailers=int(args.num_retailers),
        warehouse_holding_cost=float(args.warehouse_holding_cost),
        retailer_holding_cost=float(args.retailer_holding_cost),
        warehouse_expedited_cost=float(args.warehouse_expedited_cost),
        warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
        expedited_service_prob=float(args.expedited_service_prob),
        warehouse_capacity=int(args.warehouse_capacity),
        warehouse_inventory_cap=int(args.warehouse_inventory_cap),
        retailer_inventory_cap=int(args.retailer_inventory_cap),
        inventory_dynamics_mode=str(args.inventory_dynamics_mode),
        demand_distribution=str(args.demand_distribution),
        demand_mean=float(args.multi_demand_mean),
        demand_std=float(args.multi_demand_std),
        horizon=int(horizon),
        replications=int(replications),
        seed=int(seed),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.0)),
        objective="average_cost_after_warmup",
    )
    repro = float(dict(result["best_result"])["mean_cost"])
    return {
        "published_levels": levels,
        "published_cost": float(published),
        "repro_cost": repro,
        "gap_pct": 100.0 * (repro - float(published)) / float(published),
    }


def best_constant_base_stock_over_operating_region(base_args, budget, seed):
    """Best constant base-stock over a grid spanning the physical operating region (not the
    Gijs reduced {50..100} grid, which starves the warehouse). This is the benchmark the
    learned policy must beat."""
    # Ceilings chosen to comfortably exceed the operating region for these settings (published
    # Van Roy levels ~330/460; in-env optima ~300/525) so the benchmark grid does not bind at
    # its top edge. Raise these if the best (yw, yr) lands at a ceiling.
    warehouse_top = min(int(base_args.warehouse_inventory_cap), 700)
    retailer_top = min(int(base_args.retailer_inventory_cap), 60)
    bench_args = _are.build_reference_args(base_args.reference_name)
    bench_args.warehouse_base_stock_levels = list(range(0, warehouse_top + 1, 25))
    bench_args.retailer_base_stock_levels = list(range(0, retailer_top + 1, 5))
    return _are.best_constant_base_stock_baseline(
        bench_args, horizon=budget["eval_horizon"], replications=budget["eval_seeds"], seed=seed
    )


def _train_with_floor(args):
    """ADDITIVE/REVERSIBLE (training-path audit 2026-06-06): mirror the OWMR honest
    floor on the multi_echelon training path WITHOUT touching the shared
    invman.experiment_runner.run_experiment (used by every problem) or invman.es_mp.

    run_experiment trains via es_mp.train and DISCARDS the optimizer, so only the
    CMA-ES xbest endpoint (es.best_param(), the historically deployed individual that
    overfits the small training-seed batch) is ever evaluated. Here we replicate
    run_experiment's exact pre-train steps (cpu limits, apply_policy_name, global
    seeding, build_model) so the trained xbest is BIT-IDENTICAL to the run_experiment
    path, then call es_mp.train(return_optimizer=True) to also read xfavorite
    (es.current_param() = the CMA-ES distribution mean = result[5]). BOTH endpoints
    are scored on the SAME eval seeds via the SAME evaluate_model, and we deploy the
    cheaper one (the downside-safe best-of {xbest, xfavorite}).

    Returns (deployed_summary, xbest_summary, xfavorite_summary, deployed_endpoint,
    policy_architecture). deployed_summary is the dict evaluate_model returns
    (mean_cost / std_cost / ...).
    """
    # Same pre-train preparation run_experiment performs (so xbest is reproducible).
    normalize_args_cpu_limits(args)
    apply_policy_name(args)
    policy_spec = get_policy_spec(args)
    policy_architecture = policy_spec.architecture_label(getattr(args, "state_features", "canonical"))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trained_models_dir).mkdir(parents=True, exist_ok=True)
    set_global_seeds(getattr(args, "seed", 0))
    model = build_model(args)
    trained_model, _hist, es = _es_train(
        model=model,
        get_model_fitness=rollout_fitness.get_model_fitness,
        get_population_fitness=rollout_fitness.get_population_fitness,
        args=args,
        same_seed=getattr(args, "same_seed", False),
        limit_env_time=getattr(args, "dynamic_horizon", False),
        min_steps=getattr(args, "min_dynamic_horizon", 100),
        max_steps=getattr(args, "max_dynamic_horizon", 5000),
        return_optimizer=True,
    )
    # xbest = es.best_param() already applied inside es_mp.train (trained_model).
    xbest_summary = evaluate_model(trained_model, args)
    # xfavorite = es.current_param() = CMA-ES distribution mean (result[5]).
    xfavorite_flat = np.asarray(es.current_param(), dtype=np.float32)
    xfavorite_model = model.set_model_params(xfavorite_flat.tolist())
    xfavorite_summary = evaluate_model(xfavorite_model, args)
    if float(xfavorite_summary["mean_cost"]) < float(xbest_summary["mean_cost"]):
        deployed_summary, deployed_endpoint = xfavorite_summary, "xfavorite"
    else:
        deployed_summary, deployed_endpoint = xbest_summary, "xbest"
    return deployed_summary, xbest_summary, xfavorite_summary, deployed_endpoint, policy_architecture


def train_one(reference_name, design, depth, budget, parsed, out_dir, deploy_endpoint="floor"):
    args = _are.build_reference_args(reference_name)
    args.multi_action_design = design
    args.policy_name = make_soft_tree_policy_name(
        depth=depth, temperature=parsed.temperature, split_type="oblique", leaf_type="linear"
    )
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]
    args.experiment_name = f"{reference_name}_{design}_d{depth}"
    args.results_dir = str(out_dir / "results")
    args.log_dir = str(out_dir / "logs")
    args.trained_models_dir = str(out_dir / "models")
    t0 = time.time()
    if deploy_endpoint == "xbest":
        # Historical path: run_experiment deploys CMA-ES xbest. Reproduced EXACTLY.
        with contextlib.redirect_stdout(io.StringIO()):
            payload, results_path = run_experiment(args)
        learned = payload["evaluation"]["learned_policy"]
        return {
            "design": design,
            "depth": depth,
            "policy_name": args.policy_name,
            "policy_architecture": payload["policy_architecture"],
            "mean_cost": float(learned["mean_cost"]),
            "std_cost": float(learned["std_cost"]),
            "deployed_endpoint": "xbest",
            "xbest_cost": float(learned["mean_cost"]),
            "xfavorite_cost": None,
            "results_json": str(results_path),
            "train_seconds": round(time.time() - t0, 1),
        }
    # Default "floor": best-of {xbest, xfavorite}, downside-safe (never worse than xbest).
    with contextlib.redirect_stdout(io.StringIO()):
        deployed, xbest_s, xfav_s, deployed_endpoint, policy_architecture = _train_with_floor(args)
    return {
        "design": design,
        "depth": depth,
        "policy_name": args.policy_name,
        "policy_architecture": policy_architecture,
        "mean_cost": float(deployed["mean_cost"]),
        "std_cost": float(deployed["std_cost"]),
        "deployed_endpoint": deployed_endpoint,
        "xbest_cost": float(xbest_s["mean_cost"]),
        "xfavorite_cost": float(xfav_s["mean_cost"]),
        "results_json": None,
        "train_seconds": round(time.time() - t0, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="gijsbrechts2022_setting1")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="full")
    parser.add_argument("--designs", default="grid,direct_level",
                        help="comma-separated action designs to sweep (grid, direct_level)")
    parser.add_argument("--depths", default="2,3", help="comma-separated soft-tree depths to sweep")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--sigma_init", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    return parser.parse_args()


def main():
    parsed = parse_args()
    budget = BUDGETS[parsed.budget]
    reference_name = parsed.reference
    depths = [int(d) for d in str(parsed.depths).split(",") if d.strip()]
    out_dir = PACKAGE_ROOT / "outputs" / "multi_echelon" / f"{reference_name}_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = _are.build_reference_args(reference_name)
    print(f"=== {reference_name}  (mode={base_args.inventory_dynamics_mode}, mu={base_args.multi_demand_mean}, "
          f"sigma={base_args.multi_demand_std}, lw={base_args.warehouse_lead_time}, "
          f"lr={base_args.retailer_lead_time}, K={base_args.num_retailers}) ===")

    repro = literature_reproduction(base_args, reference_name, budget["eval_horizon"], budget["eval_seeds"], parsed.seed)
    if repro is not None:
        print(f"[literature] published constant base-stock {tuple(repro['published_levels'])} = "
              f"{repro['published_cost']} ; repo = {repro['repro_cost']:.3f} (gap {repro['gap_pct']:+.2f}%)")
    else:
        print("[literature] faithful gijs_2022 instance: no published absolute cost attached here "
              "(it reproduces on the sibling van_roy1997_* instance).")

    baseline = best_constant_base_stock_over_operating_region(base_args, budget, parsed.seed)
    print(f"[benchmark] best constant base-stock (operating-region grid): yw={baseline['warehouse_level']} "
          f"yr={baseline['retailer_level']} mean_cost={baseline['mean_cost']:.3f} +/- {baseline['std_cost']:.3f}")

    designs = [d.strip() for d in str(parsed.designs).split(",") if d.strip()]
    a3c = published_a3c_savings(reference_name)
    runs = []
    for design in designs:
        for depth in depths:
            run = train_one(reference_name, design, depth, budget, parsed, out_dir)
            runs.append(run)
            gap = 100.0 * (run["mean_cost"] - baseline["mean_cost"]) / baseline["mean_cost"]
            print(f"[train] {design:13s} d{depth} -> mean_cost={run['mean_cost']:.3f} "
                  f"+/- {run['std_cost']:.3f}  (vs best base-stock {gap:+.2f}%, {run['train_seconds']}s)")

    best = min(runs, key=lambda r: r["mean_cost"])
    best_savings_pct = 100.0 * (baseline["mean_cost"] - best["mean_cost"]) / baseline["mean_cost"]
    report = {
        "reference": reference_name,
        "budget": parsed.budget,
        "seed": parsed.seed,
        "literature_reproduction": repro,
        "benchmark_best_constant_base_stock": baseline,
        "published_a3c_savings_pct": a3c,
        "runs": runs,
        "best_run": best,
        "best_learned_savings_vs_constant_base_stock_pct": best_savings_pct,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[best] design={best['design']} depth={best['depth']} mean_cost={best['mean_cost']:.3f} "
          f"-> {best_savings_pct:+.2f}% vs best constant base-stock"
          + (f"  (published A3C savings {a3c}%)" if a3c is not None else ""))
    print(f"       model under {out_dir / 'models'}; report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
