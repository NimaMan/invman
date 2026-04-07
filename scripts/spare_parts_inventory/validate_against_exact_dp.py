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
    base_stock_params,
    dumps_json,
    ensure_parent,
    evaluate_heuristic_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    get_primary_reference,
    lead_time_mean_cover_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate spare_parts_inventory against the repo exact DP verifier and report heuristic performance on the primary reference instance."
    )
    parser.add_argument("--simulation_replications", type=int, default=256)
    parser.add_argument("--simulation_seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation() -> dict:
    return {
        "verification_reference": get_exact_verification_reference(),
        "exact_summary": get_exact_dp_summary(),
    }


def _primary_simulation_validation(replications: int, seed: int) -> dict:
    reference = get_primary_reference()
    base_stock = evaluate_heuristic_policy(
        reference,
        "base_stock",
        replications=replications,
        seed=seed,
    )
    mean_cover = evaluate_heuristic_policy(
        reference,
        "lead_time_mean_cover",
        replications=replications,
        seed=seed,
    )
    return {
        "reference": reference,
        "replications": replications,
        "seed": seed,
        "heuristics": {
            "base_stock": base_stock,
            "lead_time_mean_cover": mean_cover,
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
        f"| `base_stock_discounted_cost` | `{exact_summary['base_stock_discounted_cost']:.6f}` |",
        f"| `lead_time_mean_cover_discounted_cost` | `{exact_summary['lead_time_mean_cover_discounted_cost']:.6f}` |",
        f"| `matches_expected_optimal_discounted_cost` | `{exact_summary['matches_expected_optimal_discounted_cost']}` |",
        f"| `matches_expected_optimal_first_action` | `{exact_summary['matches_expected_optimal_first_action']}` |",
        "",
        "| Policy | Params | Mean Discounted Cost | Std | Note |",
        "| --- | --- | ---: | ---: | --- |",
        f"| `base_stock` | `{base_stock_params(primary['reference'])}` | `{primary['heuristics']['base_stock']['mean_discounted_cost']:.3f}` | `{primary['heuristics']['base_stock']['std_discounted_cost']:.3f}` | repo-native primary reference; not literature-verified |",
        f"| `lead_time_mean_cover` | `{lead_time_mean_cover_params(primary['reference'])}` | `{primary['heuristics']['lead_time_mean_cover']['mean_discounted_cost']:.3f}` | `{primary['heuristics']['lead_time_mean_cover']['std_discounted_cost']:.3f}` | repo-native primary reference; not literature-verified |",
    ]
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation(),
        "primary_simulation_validation": _primary_simulation_validation(
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
