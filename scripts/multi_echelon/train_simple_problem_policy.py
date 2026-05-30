"""Design + train a policy for the SIMPLE Van Roy multi-echelon problem.

Objective
---------
Validate the policy-design-plus-CMA-ES method end to end on the smallest instance of
the divergent_special_delivery family (van_roy1997_simple_problem: one warehouse, one
retailer, lw=0, lr=1, van_roy_1997 dynamics) before scaling to the K=10 faithful
gijs_2022 settings. "Design" = the soft-tree policy class (a state-dependent base-stock
controller); we sweep a small set of tree depths as a minimal architecture search.
"Train" = optimise each architecture's parameters with CMA-ES against the average
period cost of the simulated MDP.

Algorithm
---------
1. Build the env config from the Rust reference catalog (invman_rust) via the autoresearch
   helper build_reference_args(reference). This pins the faithful env parameters and the
   discrete base-stock action grids the soft-tree decodes into (yw in {0..10},
   yr in {0..50} for the simple problem).
2. Benchmark: invman_rust.multi_echelon_search_stationary_policy enumerates the constant
   (state-independent) base-stock grid and returns the best (yw, yr) and its long-run
   average cost after warm-up. This is the policy the learned controller must beat.
3. For each tree depth d in DEPTHS:
     a. policy = soft_tree(depth=d, oblique splits, linear leaves) over the
        full_decision_state feature vector (warehouse on-hand/pipeline + retailer
        on-hand/pipeline). The linear leaves make the order-up-to levels state dependent.
     b. CMA-ES (invman.es_mp.train via run_experiment) minimises the negative average
        period cost; each fitness evaluation is a Rust rollout of length `horizon`.
     c. Evaluate the trained policy at a longer horizon over several seeds.
4. Pick the architecture with the lowest evaluation cost, persist its model and a JSON
   report, and print a comparison table (learned vs best constant base-stock vs the
   published Van Roy constant-base-stock 51.7 and best NDP 52.6).

Expected result: on this single-echelon, near-newsvendor instance constant base-stock is
near-optimal, so a good learned policy MATCHES it (it cannot meaningfully beat it). The
room to improve over constant base-stock appears at the K=10 settings, where Gijs (2022)
reports ~9-12% A3C savings.

Usage
-----
    python scripts/multi_echelon/train_simple_problem_policy.py [--budget screening|full]
"""
import argparse
import io
import contextlib
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import importlib.util

import invman_rust  # noqa: F401  (ensures the extension is importable before training)
from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name

# Reuse build_reference_args / best_constant_base_stock_baseline from the autoresearch entrypoint.
_AUTORESEARCH = PACKAGE_ROOT / "scripts" / "multi_echelon" / "autoresearch_multi_echelon.py"
_spec = importlib.util.spec_from_file_location("autoresearch_multi_echelon", _AUTORESEARCH)
_are = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_are)

REFERENCE = "van_roy1997_simple_problem"
DEPTHS = (1, 2, 3)
BUDGETS = {
    "screening": {"training_episodes": 200, "es_population": 16, "horizon": 2000, "eval_horizon": 20000, "eval_seeds": 5},
    "full": {"training_episodes": 300, "es_population": 24, "horizon": 4000, "eval_horizon": 50000, "eval_seeds": 8},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="full")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--sigma_init", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    return parser.parse_args()


def train_one(depth, budget, parsed, out_dir):
    args = _are.build_reference_args(REFERENCE)
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
    args.experiment_name = f"simple_soft_tree_d{depth}"
    args.results_dir = str(out_dir / "results")
    args.log_dir = str(out_dir / "logs")
    args.trained_models_dir = str(out_dir / "models")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        payload, results_path = run_experiment(args)
    learned = payload["evaluation"]["learned_policy"]
    return {
        "depth": depth,
        "policy_name": args.policy_name,
        "policy_architecture": payload["policy_architecture"],
        "mean_cost": float(learned["mean_cost"]),
        "std_cost": float(learned["std_cost"]),
        "results_json": str(results_path),
        "train_seconds": round(time.time() - t0, 1),
    }


def main():
    parsed = parse_args()
    budget = BUDGETS[parsed.budget]
    out_dir = PACKAGE_ROOT / "outputs" / "multi_echelon" / "simple_problem_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = _are.build_reference_args(REFERENCE)
    baseline = _are.best_constant_base_stock_baseline(
        base_args, horizon=budget["eval_horizon"], replications=budget["eval_seeds"], seed=parsed.seed
    )
    print(f"[benchmark] best constant base-stock: yw={baseline['warehouse_level']} "
          f"yr={baseline['retailer_level']} mean_cost={baseline['mean_cost']:.3f} "
          f"+/- {baseline['std_cost']:.3f}   (published 51.7, best NDP 52.6)")

    runs = []
    for depth in DEPTHS:
        run = train_one(depth, budget, parsed, out_dir)
        runs.append(run)
        gap = 100.0 * (run["mean_cost"] - baseline["mean_cost"]) / baseline["mean_cost"]
        print(f"[train] soft_tree d{depth} oblique linear -> mean_cost={run['mean_cost']:.3f} "
              f"+/- {run['std_cost']:.3f}  (gap vs best base-stock {gap:+.2f}%, {run['train_seconds']}s)")

    best = min(runs, key=lambda r: r["mean_cost"])
    report = {
        "reference": REFERENCE,
        "budget": parsed.budget,
        "seed": parsed.seed,
        "benchmark_best_constant_base_stock": baseline,
        "published_constant_base_stock": 51.7,
        "published_best_ndp": 52.6,
        "runs": runs,
        "best_run": best,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[best] {best['policy_name']} mean_cost={best['mean_cost']:.3f} "
          f"(model under {out_dir / 'models'}); report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
