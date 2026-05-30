import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust

from invman.config import get_config
from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name


BUDGETS = {
    "screening": {"training_episodes": 200, "es_population": 8, "horizon": 1000, "eval_horizon": 5000, "eval_seeds": 2},
    "full": {"training_episodes": 1000, "es_population": 10, "horizon": 2000, "eval_horizon": 10000, "eval_seeds": 3},
}


def build_reference_args(reference_name: str):
    """Build a config Namespace for the multi-echelon problem from a Rust reference
    instance. The faithful search targets are gijsbrechts2022_setting1/2 (gijs_2022
    dynamics, Table-3 demand mean); the van_roy1997_* instances are reproduction-only.

    This reads the env parameters directly from the built `invman_rust` catalog (the
    source of truth) rather than a stale `invman.problems.*` module, and sets the action
    grids that the soft-tree policy builder turns into the discrete base-stock action space.
    """
    args = get_config([])
    reference = dict(invman_rust.multi_echelon_get_reference_instance(str(reference_name)))
    args.problem = "multi_echelon"
    args.warehouse_lead_time = int(reference["warehouse_lead_time"])
    args.retailer_lead_time = int(reference["retailer_lead_time"])
    args.num_retailers = int(reference["num_retailers"])
    args.warehouse_holding_cost = float(reference["warehouse_holding_cost"])
    args.retailer_holding_cost = float(reference["retailer_holding_cost"])
    args.warehouse_expedited_cost = float(reference["warehouse_expedited_cost"])
    args.warehouse_lost_sale_cost = float(reference["warehouse_lost_sale_cost"])
    args.expedited_service_prob = float(reference["expedited_service_prob"])
    args.warehouse_capacity = int(reference["warehouse_capacity"])
    args.warehouse_inventory_cap = int(reference["warehouse_inventory_cap"])
    args.retailer_inventory_cap = int(reference["retailer_inventory_cap"])
    args.multi_demand_mean = float(reference["demand_mean"])
    args.multi_demand_std = float(reference["demand_std"])
    args.inventory_dynamics_mode = str(reference["inventory_dynamics_mode"])
    args.demand_distribution = str(reference["demand_distribution"])
    args.warm_up_periods_ratio = float(reference.get("warm_up_periods_ratio", 0.0))
    # Long-run average cost after warm-up keeps the learned-policy fitness on the same
    # scale as the constant-base-stock benchmark used in the results table.
    args.rollout_objective = str(reference.get("rollout_objective", "average_cost_after_warmup"))
    # Discrete base-stock action grids consumed by the soft-tree policy builder.
    args.warehouse_base_stock_levels = [int(v) for v in reference["benchmark_warehouse_levels"]]
    args.retailer_base_stock_levels = [int(v) for v in reference["benchmark_retailer_levels"]]
    args.reference_name = str(reference_name)
    return args


def best_constant_base_stock_baseline(args, horizon: int, replications: int, seed: int) -> dict:
    """Best constant base-stock cost over the reference action grid, computed in Rust.

    run_experiment now returns an empty heuristics dict (the Python heuristics were
    removed in the package cleanup), so the benchmark the learned policy is compared
    against is computed here against the same env config the policy is trained on.
    """
    result = invman_rust.multi_echelon_search_stationary_policy(
        policy_kind="constant_base_stock",
        allocation_mode="min_shortage",
        warehouse_levels=list(args.warehouse_base_stock_levels),
        retailer_levels=list(args.retailer_base_stock_levels),
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
    best = dict(result["best_result"])
    return {
        "warehouse_level": int(best["warehouse_level"]),
        "retailer_level": int(best["retailer_level"]),
        "mean_cost": float(best["mean_cost"]),
        "std_cost": float(best["cost_std"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch-style loop for the multi-echelon benchmark.")
    parser.add_argument("--run_tag", default="multi_echelon_autoresearch")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    parser.add_argument("--description", required=True)
    parser.add_argument("--reference", default="gijsbrechts2022_setting2")
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--tree_leaf_type", choices=["constant", "linear"], default="linear")
    parser.add_argument("--sigma_init", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    return parser.parse_args()


def _git_short_commit(project_root: Path) -> str:
    result = subprocess.run(["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main():
    parsed = parse_args()
    args = build_reference_args(parsed.reference)
    budget = BUDGETS[parsed.budget]
    args.problem = "multi_echelon"
    args.policy_name = make_soft_tree_policy_name(
        depth=parsed.tree_depth,
        temperature=parsed.tree_temperature,
        split_type=parsed.tree_split_type,
        leaf_type=parsed.tree_leaf_type,
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
    args.experiment_name = f"{parsed.run_tag}_{parsed.budget}_{args.policy_name}"

    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    results_tsv = root / "results.tsv"
    root.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["commit", "experiment_name", "reference", "budget", "policy_architecture", "mean_cost", "best_heuristic", "heuristic_gap", "description"])

    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    payload, results_path = run_experiment(args)
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    heuristics = payload["evaluation"].get("heuristics", {})
    if heuristics:
        best_heuristic_cost = min(value["mean_cost"] for value in heuristics.values())
    else:
        # run_experiment returns no Python-side heuristic baselines; compute the best
        # constant base-stock cost in Rust against the same (faithful) env config.
        baseline = best_constant_base_stock_baseline(
            args, budget["eval_horizon"], budget["eval_seeds"], parsed.seed
        )
        best_heuristic_cost = baseline["mean_cost"]
        payload["evaluation"]["heuristics"] = {"constant_base_stock": baseline}
    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            _git_short_commit(PACKAGE_ROOT),
            args.experiment_name,
            parsed.reference,
            parsed.budget,
            payload["policy_architecture"],
            f"{learned_cost:.6f}",
            f"{best_heuristic_cost:.6f}",
            f"{learned_cost - best_heuristic_cost:.6f}",
            parsed.description,
        ])
    print(json.dumps({"results_json": str(results_path), "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
