import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policies.registry import apply_policy_name
from invman.problems.lost_sales.experiment_spec import COMMON_BUDGET as VANILLA_BUDGET
from invman.problems.lost_sales.reference_instances import build_reference_args as build_vanilla_args
from invman.problems.lost_sales_fixed_order_cost.experiment_spec import (
    COMMON_BUDGET as FIXED_COST_BUDGET,
)
from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    build_reference_args as build_fixed_cost_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single raw-state policy probe on a canonical lost-sales instance."
    )
    parser.add_argument("--problem", choices=["lost_sales", "lost_sales_fixed_order_cost"], required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--policy", default="linear_direct_quantity")
    parser.add_argument("--state_features", default="raw_pipeline")
    parser.add_argument("--rollout_backend", default="python")
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    return parser.parse_args()


def build_args(parsed):
    if parsed.problem == "lost_sales":
        args = build_vanilla_args(parsed.reference)
        args.problem = "lost_sales"
        args.training_episodes = VANILLA_BUDGET["training_episodes_default"]
        args.es_population = VANILLA_BUDGET["es_population"]
        args.horizon = VANILLA_BUDGET["horizon_default"]
        args.sigma_init = VANILLA_BUDGET["sigma_init"]
        args.save_every = VANILLA_BUDGET["save_every"]
    else:
        args = build_fixed_cost_args(parsed.reference)
        args.problem = "lost_sales_fixed_order_cost"
        args.training_episodes = FIXED_COST_BUDGET["training_episodes"]
        args.es_population = FIXED_COST_BUDGET["es_population"]
        args.horizon = FIXED_COST_BUDGET["horizon"]
        args.dynamic_horizon = FIXED_COST_BUDGET["dynamic_horizon"]
        args.min_dynamic_horizon = FIXED_COST_BUDGET["min_dynamic_horizon"]
        args.max_dynamic_horizon = FIXED_COST_BUDGET["max_dynamic_horizon"]
        args.sigma_init = FIXED_COST_BUDGET["sigma_init"]

    root = PACKAGE_ROOT / "outputs" / "benchmarks" / parsed.run_tag
    for dirname in ("results", "logs", "models"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    args.reference_instance = parsed.reference
    args.seed = parsed.seed
    args.same_seed = False
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.policy_name = parsed.policy
    args.state_features = parsed.state_features
    apply_policy_name(args)
    args.rollout_backend = parsed.rollout_backend
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{parsed.run_tag}_{parsed.reference}_{parsed.policy}"
    return args


def main():
    parsed = parse_args()
    args = build_args(parsed)
    payload, result_path = run_experiment(args)
    print(result_path)


if __name__ == "__main__":
    main()
