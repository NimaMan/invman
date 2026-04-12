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

from common import (  # noqa: E402
    dumps_json,
    ensure_parent,
    evaluate_stationary_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    list_references,
    savings_pct,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Rust multi-echelon package against the repo exact verifier and the published Van Roy constant-base-stock rows."
    )
    parser.add_argument("--van_roy_replications", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation() -> dict:
    exact_reference = get_exact_verification_reference()
    exact_summary = get_exact_dp_summary()
    best_stationary_cost = min(
        float(exact_summary["sequential_discounted_cost"]),
        float(exact_summary["proportional_discounted_cost"]),
        float(exact_summary["min_shortage_discounted_cost"]),
    )
    return {
        "verification_reference": exact_reference,
        "exact_summary": exact_summary,
        "best_stationary_discounted_cost": best_stationary_cost,
        "optimal_gap_vs_best_stationary": float(exact_summary["optimal_discounted_cost"])
        - best_stationary_cost,
        "optimal_beats_best_stationary": float(exact_summary["optimal_discounted_cost"])
        <= best_stationary_cost + 1e-12,
    }


def _van_roy_absolute_validation(replications: int, seed: int) -> list[dict]:
    rows = []
    for reference in list_references():
        published_cost = reference.get("published_constant_base_stock_mean_cost")
        if published_cost is None:
            continue
        published_levels = [int(value) for value in reference["published_constant_base_stock_levels"]]
        repo_eval = evaluate_stationary_policy(
            reference,
            warehouse_level=published_levels[0],
            retailer_level=published_levels[1],
            allocation_mode=str(reference["policy_allocation_mode"]),
            policy_kind="regular_base_stock",
            replications=replications,
            seed=seed,
        )
        rows.append(
            {
                "reference": str(reference["name"]),
                "literature_verified": bool(reference["literature_verified"]),
                "published_constant_base_stock_levels": published_levels,
                "published_constant_base_stock_mean_cost": float(published_cost),
                "repo_reproduced_constant_base_stock_mean_cost": float(repo_eval["mean_cost"]),
                "repo_reproduced_constant_base_stock_std": float(repo_eval["cost_std"]),
                "gap_vs_published_constant_base_stock": float(
                    repo_eval["mean_cost"] - float(published_cost)
                ),
                "published_van_roy_best_ndp_mean_cost": (
                    float(reference["published_van_roy_best_ndp_mean_cost"])
                    if reference.get("published_van_roy_best_ndp_mean_cost") is not None
                    else None
                ),
                "published_van_roy_best_ndp_savings_pct": (
                    savings_pct(float(published_cost), float(reference["published_van_roy_best_ndp_mean_cost"]))
                    if reference.get("published_van_roy_best_ndp_mean_cost") is not None
                    else None
                ),
                "published_a3c_savings_pct": reference.get("published_a3c_savings_pct"),
                "protocol": {
                    "replications": int(replications),
                    "periods": int(reference["benchmark_periods"]),
                    "allocation_mode": str(reference["policy_allocation_mode"]),
                },
            }
        )
    return rows


def _markdown(payload: dict) -> str:
    exact = payload["exact_validation"]
    lines = [
        "| Exact Verification Metric | Value |",
        "| --- | --- |",
        f"| `optimal_discounted_cost` | `{exact['exact_summary']['optimal_discounted_cost']:.12f}` |",
        f"| `optimal_first_action` | `{exact['exact_summary']['optimal_first_action']}` |",
        f"| `best_stationary_discounted_cost` | `{exact['best_stationary_discounted_cost']:.12f}` |",
        f"| `optimal_gap_vs_best_stationary` | `{exact['optimal_gap_vs_best_stationary']:.12f}` |",
        f"| `optimal_beats_best_stationary` | `{exact['optimal_beats_best_stationary']}` |",
        "",
        "| Van Roy Reference | Published Heuristic Levels | Repo Heuristic Cost | Published Heuristic Cost | Gap | Published Best NDP |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["van_roy_absolute_validation"]:
        ndp = (
            "na"
            if row["published_van_roy_best_ndp_mean_cost"] is None
            else f"{row['published_van_roy_best_ndp_mean_cost']:.3f}"
        )
        lines.append(
            f"| `{row['reference']}` | `{row['published_constant_base_stock_levels']}` | "
            f"`{row['repo_reproduced_constant_base_stock_mean_cost']:.3f}` | "
            f"`{row['published_constant_base_stock_mean_cost']:.3f}` | "
            f"`{row['gap_vs_published_constant_base_stock']:.3f}` | `{ndp}` |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation(),
        "van_roy_absolute_validation": _van_roy_absolute_validation(
            replications=parsed.van_roy_replications,
            seed=parsed.seed,
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
