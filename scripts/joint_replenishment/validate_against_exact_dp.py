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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Rust joint_replenishment implementation against the repo exact DP verifier. The carried literature settings come from Vanvuchelen et al. Table 2, but the public paper does not expose exact per-setting assertion rows."
    )
    parser.add_argument("--simulation_replications", type=int, default=512)
    parser.add_argument("--simulation_seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation() -> dict:
    exact_reference = get_exact_verification_reference()
    exact_summary = get_exact_dp_summary()
    return {
        "verification_reference": exact_reference,
        "exact_summary": exact_summary,
    }


def _simulation_validation(replications: int, seed: int) -> dict:
    reference = get_exact_verification_reference()
    moq = evaluate_heuristic_policy(
        reference,
        "minimum_order_quantity",
        list(reference["moq_item_targets"]) + [
            int(reference["moq_review_period"]),
            float(reference["moq_rounding_threshold"]),
        ],
        replications=replications,
        seed=seed,
    )
    dynout = evaluate_heuristic_policy(
        reference,
        "dynamic_order_up_to",
        list(reference["dynout_item_targets"]),
        replications=replications,
        seed=seed,
    )
    return {
        "replications": int(replications),
        "seed": int(seed),
        "heuristics": {
            "minimum_order_quantity": moq,
            "dynamic_order_up_to": dynout,
        },
    }


def _markdown(payload: dict) -> str:
    exact_summary = payload["exact_validation"]["exact_summary"]
    simulation = payload["simulation_validation"]
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
        f"| `moq_discounted_cost` | `{exact_summary['moq_discounted_cost']:.6f}` |",
        f"| `dynout_discounted_cost` | `{exact_summary['dynout_discounted_cost']:.6f}` |",
        "",
        "| Policy | Mean Discounted Cost | Std | Note |",
        "| --- | ---: | ---: | --- |",
        f"| `minimum_order_quantity` | `{simulation['heuristics']['minimum_order_quantity']['mean_cost']:.3f}` | `{simulation['heuristics']['minimum_order_quantity']['cost_std']:.3f}` | Monte Carlo check around exact heuristic benchmark |",
        f"| `dynamic_order_up_to` | `{simulation['heuristics']['dynamic_order_up_to']['mean_cost']:.3f}` | `{simulation['heuristics']['dynamic_order_up_to']['cost_std']:.3f}` | Monte Carlo check around exact heuristic benchmark |",
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
