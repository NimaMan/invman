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
    evaluate_heuristic_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    get_primary_reference,
    interval_stock_params,
    returnability_buffer_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate procurement_removal_inventory against the repo exact DP verifier and report heuristic performance on the primary reference instance."
    )
    parser.add_argument("--simulation_seeds", type=int, default=256)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation() -> dict:
    exact_reference = get_exact_verification_reference()
    exact_summary = get_exact_dp_summary()
    return {
        "verification_reference": exact_reference,
        "exact_summary": exact_summary,
    }


def _primary_simulation_validation(num_seeds: int) -> dict:
    reference = get_primary_reference()
    seeds = list(range(123, 123 + num_seeds))
    interval = evaluate_heuristic_policy(reference, "interval_stock", seeds)
    buffer = evaluate_heuristic_policy(
        reference,
        "returnability_buffer_interval_stock",
        seeds,
    )
    return {
        "reference": reference,
        "num_seeds": num_seeds,
        "heuristics": {
            "interval_stock": interval,
            "returnability_buffer_interval_stock": buffer,
        },
    }


def _markdown(payload: dict) -> str:
    exact_summary = payload["exact_validation"]["exact_summary"]
    primary = payload["primary_simulation_validation"]
    lines = [
        "| Verification Status | Value |",
        "| --- | --- |",
        f"| `verification_source` | `{payload['exact_validation']['verification_reference']['verification_source']}` |",
        f"| `literature_verified` | `{payload['exact_validation']['verification_reference']['literature_verified']}` |",
        "",
        "| Verification Metric | Value |",
        "| --- | ---: |",
        f"| `optimal_discounted_cost` | `{exact_summary['optimal_discounted_cost']:.6f}` |",
        f"| `optimal_first_action` | `{exact_summary['optimal_first_action']}` |",
        f"| `interval_stock_discounted_cost` | `{exact_summary['interval_stock_discounted_cost']:.6f}` |",
        f"| `returnability_buffer_discounted_cost` | `{exact_summary['returnability_buffer_discounted_cost']:.6f}` |",
        "",
        "| Policy | Params | Mean Discounted Cost | Std | Note |",
        "| --- | --- | ---: | ---: | --- |",
        f"| `interval_stock` | `{interval_stock_params(primary['reference'])}` | `{primary['heuristics']['interval_stock']['mean_discounted_cost']:.3f}` | `{primary['heuristics']['interval_stock']['std_discounted_cost']:.3f}` | repo-native primary reference; not literature-verified |",
        f"| `returnability_buffer_interval_stock` | `{returnability_buffer_params(primary['reference'])}` | `{primary['heuristics']['returnability_buffer_interval_stock']['mean_discounted_cost']:.3f}` | `{primary['heuristics']['returnability_buffer_interval_stock']['std_discounted_cost']:.3f}` | repo-native primary reference; not literature-verified |",
    ]
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation(),
        "primary_simulation_validation": _primary_simulation_validation(parsed.simulation_seeds),
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
