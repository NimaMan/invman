import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import benchmark_reference_instance


def build_parser():
    parser = argparse.ArgumentParser(description="Report fixed-cost heuristic backend availability after the Rust-first migration.")
    parser.add_argument("--reference_instance", default="lit_pois_mu5_l4_p4_k5")
    parser.add_argument("--search_horizon", default=2000, type=int)
    parser.add_argument("--eval_horizon", default=50000, type=int)
    parser.add_argument("--eval_seeds", default=3, type=int)
    parser.add_argument("--position_upper_bound", default=None, type=int)
    parser.add_argument("--search_seed", default=1234, type=int)
    parser.add_argument("--modified_search_mode", default="exhaustive", choices=["guided", "exhaustive"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    shared_kwargs = {
        "search_horizon": args.search_horizon,
        "eval_horizon": args.eval_horizon,
        "eval_seeds": args.eval_seeds,
        "position_upper_bound": args.position_upper_bound,
        "search_seed": args.search_seed,
        "modified_search_mode": args.modified_search_mode,
    }

    rust_payload = benchmark_reference_instance(args.reference_instance, backend="rust", **shared_kwargs)

    comparison = {
        "reference_instance": args.reference_instance,
        "python_backend": {
            "available": False,
            "reason": "the fixed-cost Python env/search backend was removed in the Rust-first migration",
        },
        "rust_backend": rust_payload,
        "same_best_params": None,
    }
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
