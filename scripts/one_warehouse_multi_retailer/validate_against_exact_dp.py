import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    dumps_json,
    ensure_parent,
    evaluate_echelon_base_stock_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    search_best_echelon_base_stock,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Rust one_warehouse_multi_retailer implementation against the repo exact verifier and the best exact echelon base-stock heuristics."
    )
    parser.add_argument("--simulation_replications", type=int, default=2048)
    parser.add_argument("--simulation_seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation() -> dict:
    reference = get_exact_verification_reference()
    exact_summary = get_exact_dp_summary()
    best_proportional = search_best_echelon_base_stock(
        reference,
        allocation_policy="proportional",
    )
    best_min_shortage = search_best_echelon_base_stock(
        reference,
        allocation_policy="min_shortage",
    )
    return {
        "verification_reference": reference,
        "exact_summary": exact_summary,
        "best_heuristics": {
            "proportional": best_proportional,
            "min_shortage": best_min_shortage,
        },
    }


def _simulation_validation(replications: int, seed: int) -> dict:
    reference = get_exact_verification_reference()
    best_proportional = search_best_echelon_base_stock(
        reference,
        allocation_policy="proportional",
    )
    best_min_shortage = search_best_echelon_base_stock(
        reference,
        allocation_policy="min_shortage",
    )
    return {
        "replications": int(replications),
        "seed": int(seed),
        "heuristics": {
            "proportional": evaluate_echelon_base_stock_policy(
                reference,
                warehouse_base_stock_level=best_proportional["warehouse_base_stock_level"],
                retailer_base_stock_levels=best_proportional["retailer_base_stock_levels"],
                allocation_policy="proportional",
                replications=replications,
                seed=seed,
            ),
            "min_shortage": evaluate_echelon_base_stock_policy(
                reference,
                warehouse_base_stock_level=best_min_shortage["warehouse_base_stock_level"],
                retailer_base_stock_levels=best_min_shortage["retailer_base_stock_levels"],
                allocation_policy="min_shortage",
                replications=replications,
                seed=seed,
            ),
        },
    }


def _markdown(payload: dict) -> str:
    exact_summary = payload["exact_validation"]["exact_summary"]
    exact_validation = payload["exact_validation"]
    simulation = payload["simulation_validation"]
    lines = [
        "| Verification Metric | Value |",
        "| --- | --- |",
        f"| `literature_verified` | `{exact_validation['verification_reference']['literature_verified']}` |",
        f"| `optimal_discounted_cost` | `{exact_summary['optimal_discounted_cost']:.6f}` |",
        f"| `optimal_first_action` | `{exact_summary['optimal_first_action']}` |",
        "",
        "| Policy | Params | Exact Discounted Cost | Monte Carlo Mean Cost | Monte Carlo Std |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| `best_echelon_base_stock_proportional` | `[{exact_validation['best_heuristics']['proportional']['warehouse_base_stock_level']}, {', '.join(str(v) for v in exact_validation['best_heuristics']['proportional']['retailer_base_stock_levels'])}]` | `{exact_validation['best_heuristics']['proportional']['mean_cost']:.6f}` | `{simulation['heuristics']['proportional']['mean_cost']:.6f}` | `{simulation['heuristics']['proportional']['cost_std']:.6f}` |",
        f"| `best_echelon_base_stock_min_shortage` | `[{exact_validation['best_heuristics']['min_shortage']['warehouse_base_stock_level']}, {', '.join(str(v) for v in exact_validation['best_heuristics']['min_shortage']['retailer_base_stock_levels'])}]` | `{exact_validation['best_heuristics']['min_shortage']['mean_cost']:.6f}` | `{simulation['heuristics']['min_shortage']['mean_cost']:.6f}` | `{simulation['heuristics']['min_shortage']['cost_std']:.6f}` |",
    ]
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation(),
        "simulation_validation": _simulation_validation(
            parsed.simulation_replications,
            parsed.simulation_seed,
        ),
    }
    payload["markdown"] = _markdown(payload)

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
