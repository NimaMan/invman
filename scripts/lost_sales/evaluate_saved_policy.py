import argparse
import json
import sys
from copy import copy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.policy import Policy
from invman.rollout_fitness import get_model_fitness
from scripts.lost_sales.benchmark_canonical_suite import (
    build_reference_args as build_lost_sales_reference_args,
)
from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import (
    build_reference_args as build_fixed_cost_reference_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved learned policy on a named reference instance.")
    parser.add_argument("--model_dir", required=True, help="Directory containing policy_artifact.json and model_params.npy.")
    parser.add_argument("--problem", choices=["lost_sales", "lost_sales_fixed_order_cost"], required=True)
    parser.add_argument("--reference", required=True, help="Named reference instance for the selected problem.")
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--track_demand",
        action="store_true",
        help="Accepted for old CLI compatibility; current evaluation is Rust-backed.",
    )
    return parser.parse_args()


def build_reference_args(problem: str, reference: str):
    if problem == "lost_sales":
        return build_lost_sales_reference_args(reference)
    if problem == "lost_sales_fixed_order_cost":
        return build_fixed_cost_reference_args(reference)
    raise NotImplementedError(f"Unsupported problem '{problem}'")


def summarize_costs(costs):
    import numpy as np

    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(len(costs)),
    }


def main():
    parsed = parse_args()
    model = Policy.load(parsed.model_dir)
    args = build_reference_args(parsed.problem, parsed.reference)
    args.problem = parsed.problem
    args.reference_instance = parsed.reference
    args.horizon = parsed.eval_horizon
    args.rollout_backend = "rust"

    costs = []
    for seed_offset in range(parsed.eval_seeds):
        eval_args = copy(args)
        seed = parsed.seed + seed_offset
        reward, _ = get_model_fitness(
            model,
            eval_args,
            seed=seed,
            track_demand=parsed.track_demand,
        )
        costs.append(-float(reward))

    payload = {
        "model_dir": str(Path(parsed.model_dir).resolve()),
        "problem": parsed.problem,
        "reference": parsed.reference,
        "eval_horizon": parsed.eval_horizon,
        "eval_seeds": parsed.eval_seeds,
        "rollout_backend": "rust",
        "evaluation": summarize_costs(costs),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
