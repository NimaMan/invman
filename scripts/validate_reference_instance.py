import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.lost_sales.reference_instances import (
    VANILLA_L4_P4_POISSON5,
    evaluate_cap_sensitivity,
    evaluate_reference_heuristics,
    get_reference_instance,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Validate the cleaned lost-sales code on a reference instance.")
    parser.add_argument("--reference", default=VANILLA_L4_P4_POISSON5.name, help="Reference instance name.")
    parser.add_argument("--horizon", default=int(1e5), type=int, help="Simulation horizon per run.")
    parser.add_argument("--seed", default=123, type=int, help="Initial seed.")
    parser.add_argument("--num_seeds", default=3, type=int, help="Number of sequential seeds.")
    parser.add_argument("--tolerance", default=None, type=float, help="Override tolerance for pass/fail.")
    parser.add_argument("--caps", nargs="+", type=int, default=None, help="Action caps used for sensitivity checks.")
    return parser.parse_args()


def main():
    args = parse_args()
    instance = get_reference_instance(args.reference)
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    tolerance = instance.tolerance if args.tolerance is None else args.tolerance

    heuristic_results = evaluate_reference_heuristics(
        name=args.reference,
        horizon=args.horizon,
        seeds=seeds,
    )
    cap_results = evaluate_cap_sensitivity(
        name=args.reference,
        caps=args.caps,
        seed=args.seed,
        horizon=args.horizon,
    )

    payload = {
        "reference": args.reference,
        "params": instance.params,
        "expected_costs": instance.expected_costs,
        "heuristic_results": heuristic_results,
        "cap_sensitivity": cap_results,
        "passes_reference_check": {
            heuristic_name: heuristic_summary["abs_gap"] <= tolerance
            for heuristic_name, heuristic_summary in heuristic_results.items()
        },
        "passes_ordering_check": (
            heuristic_results["myopic2"]["mean_cost"]
            < heuristic_results["myopic1"]["mean_cost"]
            < heuristic_results["svbs"]["mean_cost"]
        ),
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
