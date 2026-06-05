import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import (
    benchmark_reference_instance,
    get_benchmark_grid,
)


def benchmark_grid(grid_name, limit=None, **kwargs):
    grid = get_benchmark_grid(grid_name)
    instances = grid["instances"][:limit]
    return {
        "grid_name": grid_name,
        "num_instances": len(instances),
        "instances": [
            benchmark_reference_instance(instance["name"], **kwargs)
            for instance in instances
        ],
        "note": "fixed-cost heuristic grid baselines are evaluated through the Rust fixed-cost heuristic search binding",
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Report fixed-cost heuristic-baseline availability on the Rust experiment grid."
    )
    parser.add_argument(
        "--grid_name",
        default="lost_sales_style_full_grid_mu5",
        help="Named benchmark grid from the fixed-order-cost Rust binding.",
    )
    parser.add_argument("--limit", default=None, type=int, help="Limit the number of instances for smoke tests.")
    parser.add_argument("--search_horizon", default=None, type=int, help="Override the search horizon.")
    parser.add_argument("--eval_horizon", default=None, type=int, help="Override the evaluation horizon.")
    parser.add_argument("--eval_seeds", default=None, type=int, help="Override the number of evaluation seeds.")
    parser.add_argument("--position_upper_bound", default=None, type=int, help="Override the search upper bound for s and S.")
    parser.add_argument("--search_seed", default=None, type=int, help="Override the search seed.")
    parser.add_argument("--top_k_s_s_pairs", default=None, type=int, help="Override the number of s,S pairs used for the modified search.")
    parser.add_argument("--q_window", default=None, type=int, help="Override the q search window around the paper heuristic.")
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    payload = benchmark_grid(
        grid_name=cli_args.grid_name,
        limit=cli_args.limit,
        search_horizon=cli_args.search_horizon,
        eval_horizon=cli_args.eval_horizon,
        eval_seeds=cli_args.eval_seeds,
        position_upper_bound=cli_args.position_upper_bound,
        search_seed=cli_args.search_seed,
        top_k_s_s_pairs=cli_args.top_k_s_s_pairs,
        q_window=cli_args.q_window,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
