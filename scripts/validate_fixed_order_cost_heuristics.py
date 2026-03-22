import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.lost_sales_fixed_order_cost.benchmark import benchmark_reference_instance


def build_parser():
    parser = argparse.ArgumentParser(description="Validate fixed-order-cost heuristics on a starter instance.")
    parser.add_argument(
        "--reference_instance",
        default="lit_pois_mu5_l4_p4_k5",
        help="Named reference instance from the fixed-order-cost problem package.",
    )
    parser.add_argument("--search_horizon", default=None, type=int, help="Override the search horizon.")
    parser.add_argument("--eval_horizon", default=None, type=int, help="Override the evaluation horizon.")
    parser.add_argument("--eval_seeds", default=None, type=int, help="Override the number of evaluation seeds.")
    parser.add_argument("--position_upper_bound", default=None, type=int, help="Override the search upper bound for s and S.")
    parser.add_argument("--search_seed", default=None, type=int, help="Override the search seed.")
    parser.add_argument("--top_k_s_s_pairs", default=None, type=int, help="Override the number of s,S pairs used for the modified search.")
    parser.add_argument("--q_window", default=None, type=int, help="Override the q search window around the paper heuristic.")
    parser.add_argument("--backend", default="python", choices=["python", "rust"], help="Search backend for heuristic parameter search.")
    parser.add_argument(
        "--modified_search_mode",
        default="guided",
        choices=["guided", "exhaustive"],
        help="Search mode used for the modified (s,S,q) policy.",
    )
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    payload = benchmark_reference_instance(
        cli_args.reference_instance,
        search_horizon=cli_args.search_horizon,
        eval_horizon=cli_args.eval_horizon,
        eval_seeds=cli_args.eval_seeds,
        position_upper_bound=cli_args.position_upper_bound,
        search_seed=cli_args.search_seed,
        top_k_s_s_pairs=cli_args.top_k_s_s_pairs,
        q_window=cli_args.q_window,
        backend=cli_args.backend,
        modified_search_mode=cli_args.modified_search_mode,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
