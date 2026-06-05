import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust


PUBLISHED_VALIDATION_REFERENCE_NAME = "bijvank2015_table1_l2_p14_k5"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate fixed-cost heuristics on the published known-optimum reference instance."
    )
    parser.add_argument(
        "--reference_instance",
        default=PUBLISHED_VALIDATION_REFERENCE_NAME,
        help="Named fixed-cost reference instance to validate against the literature anchor.",
    )
    parser.add_argument("--search_horizon", default=None, type=int, help="Override the search horizon.")
    parser.add_argument("--eval_horizon", default=None, type=int, help="Override the evaluation horizon.")
    parser.add_argument("--eval_seeds", default=None, type=int, help="Override the number of evaluation seeds.")
    parser.add_argument("--position_upper_bound", default=None, type=int, help="Override the search upper bound for s and S.")
    parser.add_argument("--search_seed", default=None, type=int, help="Override the search seed.")
    parser.add_argument("--top_k_s_s_pairs", default=None, type=int, help="Override the number of s,S pairs used for the modified search.")
    parser.add_argument("--q_window", default=None, type=int, help="Override the q search window around the paper heuristic.")
    parser.add_argument(
        "--backend",
        default="rust",
        choices=["rust"],
        help="Backend for the exact literature validation.",
    )
    parser.add_argument(
        "--modified_search_mode",
        default="guided",
        choices=["guided"],
        help="Compatibility option; the Rust validator evaluates the published policies directly.",
    )
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    inventory_position_cap = cli_args.position_upper_bound or 24
    payload = invman_rust.lost_sales_fixed_order_cost_exact_literature_summary(
        cli_args.reference_instance,
        inventory_position_cap,
    )
    ignored = {
        "search_horizon": cli_args.search_horizon,
        "eval_horizon": cli_args.eval_horizon,
        "eval_seeds": cli_args.eval_seeds,
        "search_seed": cli_args.search_seed,
        "top_k_s_s_pairs": cli_args.top_k_s_s_pairs,
        "q_window": cli_args.q_window,
    }
    ignored = {key: value for key, value in ignored.items() if value is not None}
    if ignored:
        payload["ignored_cli_options"] = ignored
        payload["note"] = (
            "The current Rust validator is an exact average-cost literature check; "
            "simulation/search horizon options are ignored."
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
