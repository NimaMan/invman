#!/usr/bin/env python3
"""Replay the canonical fixed-cost ordinal experiment with an explicit backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name
from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import build_reference_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the canonical fixed-cost ordinal experiment. The fixed-cost "
            "Python simulator has been retired; this probe now runs through Rust."
        )
    )
    parser.add_argument("--backend", choices=["rust"], default="rust")
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--reference", default="lit_pois_mu5_l4_p4_k5")
    parser.add_argument("--policy_name", default="linear_gated_ordinal_quantity")
    parser.add_argument("--state_normalizer", default="quantity_scale")
    parser.add_argument("--state_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_episodes", type=int, default=5000)
    parser.add_argument("--es_population", type=int, default=50)
    parser.add_argument("--training_horizon", type=int, default=2000)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    return parser.parse_args()


def build_args(parsed: argparse.Namespace):
    root = Path(__file__).resolve().parents[3] / "outputs" / "benchmarks" / parsed.run_tag
    for dirname in ("results", "logs", "models"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    args = build_reference_args(parsed.reference)
    args.problem = "lost_sales_fixed_order_cost"
    args.reference_instance = parsed.reference
    args.policy_name = parsed.policy_name
    args.state_normalizer = parsed.state_normalizer
    apply_policy_name(args)
    if parsed.state_scale is not None:
        args.state_scale = parsed.state_scale
    args.rollout_backend = parsed.backend
    args.training_method = "cma"
    args.parameter_optimizer = "cma"
    args.training_episodes = parsed.training_episodes
    args.es_population = parsed.es_population
    args.horizon = parsed.training_horizon
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = 5.0
    args.seed = parsed.seed
    args.same_seed = False
    args.mp_num_processors = parsed.mp_num_processors
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{parsed.run_tag}_{args.policy_name}"
    return args


def main() -> None:
    parsed = parse_args()
    args = build_args(parsed)
    payload, result_path = run_experiment(args)
    print(json.dumps({"result_path": str(result_path), "mean_cost": payload["evaluation"]["learned_policy"]["mean_cost"]}, indent=2))


if __name__ == "__main__":
    main()
