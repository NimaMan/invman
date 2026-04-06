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
    linear_inflation_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Rust random-yield inventory implementation against repo exact DP and benchmark heuristics. If public literature numbers are unavailable, the verification remains explicitly non-literature-verified."
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
    linear_inflation = evaluate_heuristic_policy(reference, "linear_inflation", seeds)
    weighted_newsvendor = evaluate_heuristic_policy(reference, "weighted_newsvendor", seeds)
    return {
        "reference": reference,
        "num_seeds": num_seeds,
        "linear_inflation_params": linear_inflation_params(reference),
        "heuristics": {
            "linear_inflation": linear_inflation,
            "weighted_newsvendor": weighted_newsvendor,
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
        f"| `linear_inflation_discounted_cost` | `{exact_summary['linear_inflation_discounted_cost']:.6f}` |",
        f"| `weighted_newsvendor_discounted_cost` | `{exact_summary['weighted_newsvendor_discounted_cost']:.6f}` |",
        f"| `matches_expected_optimal_discounted_cost` | `{exact_summary['matches_expected_optimal_discounted_cost']}` |",
        f"| `matches_expected_optimal_first_action` | `{exact_summary['matches_expected_optimal_first_action']}` |",
        "",
        "| Policy | Params | Mean Discounted Cost | Std | Note |",
        "| --- | --- | ---: | ---: | --- |",
        f"| `linear_inflation` | `{primary['linear_inflation_params']}` | `{primary['heuristics']['linear_inflation']['mean_cost']:.3f}` | `{primary['heuristics']['linear_inflation']['cost_std']:.3f}` | repo-native primary reference; not literature-verified |",
        f"| `weighted_newsvendor` | `[]` | `{primary['heuristics']['weighted_newsvendor']['mean_cost']:.3f}` | `{primary['heuristics']['weighted_newsvendor']['cost_std']:.3f}` | repo-native primary reference; not literature-verified |",
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
