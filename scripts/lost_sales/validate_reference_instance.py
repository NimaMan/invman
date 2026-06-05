import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust

DEFAULT_REFERENCE = "vanilla_l4_p4_poisson5"
DEFAULT_TOLERANCE = 0.05


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Validate the cleaned lost-sales code on a reference instance.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Reference instance name.")
    parser.add_argument("--horizon", default=int(1e5), type=int, help="Simulation horizon per run.")
    parser.add_argument("--seed", default=123, type=int, help="Initial seed.")
    parser.add_argument("--num_seeds", default=3, type=int, help="Number of sequential seeds.")
    parser.add_argument("--tolerance", default=None, type=float, help="Override tolerance for pass/fail.")
    parser.add_argument("--caps", nargs="+", type=int, default=None, help="Action caps used for sensitivity checks.")
    return parser.parse_args()


def _heuristic_cost(ref: dict, heuristic: str, *, horizon: int, seed: int) -> float:
    return float(
        invman_rust.lost_sales_heuristic_mean_cost(
            heuristic=heuristic,
            demand_kind=str(ref["demand_kind"]),
            demand_rate=float(ref["demand_rate"]),
            demand_lambda_low=float(ref.get("demand_lambda_low", 0.0)),
            demand_lambda_high=float(ref.get("demand_lambda_high", 0.0)),
            demand_p00=float(ref.get("demand_p00", 0.0)),
            demand_p11=float(ref.get("demand_p11", 0.0)),
            lead_time=int(ref["lead_time"]),
            holding_cost=float(ref["holding_cost"]),
            shortage_cost=float(ref["shortage_cost"]),
            procurement_cost=0.0,
            fixed_order_cost=0.0,
            horizon=int(horizon),
            seed=int(seed),
            warm_up_periods_ratio=0.2,
            order_search_upper_bound=200,
            heuristic_discount_factor=0.995,
        )
    )


def main():
    args = parse_args()
    ref = invman_rust.lost_sales_reference_costs(args.reference)
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    tolerance = DEFAULT_TOLERANCE if args.tolerance is None else args.tolerance

    heuristic_results = {}
    for heuristic in ("myopic1", "myopic2", "svbs"):
        values = [_heuristic_cost(ref, heuristic, horizon=args.horizon, seed=seed) for seed in seeds]
        expected = ref["costs"].get(heuristic)
        mean_cost = sum(values) / len(values)
        heuristic_results[heuristic] = {
            "mean_cost": mean_cost,
            "expected_cost": expected,
            "abs_gap": None if expected is None else abs(mean_cost - float(expected)),
            "seed_costs": values,
        }

    payload = {
        "reference": args.reference,
        "params": {
            key: ref[key]
            for key in (
                "demand_kind",
                "demand_rate",
                "demand_lambda_low",
                "demand_lambda_high",
                "demand_p00",
                "demand_p11",
                "lead_time",
                "holding_cost",
                "shortage_cost",
            )
        },
        "expected_costs": ref["costs"],
        "heuristic_results": heuristic_results,
        "cap_sensitivity": "not implemented in the Rust-routed validator",
        "passes_reference_check": {
            heuristic_name: (
                heuristic_summary["abs_gap"] is not None
                and heuristic_summary["abs_gap"] <= tolerance
            )
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
