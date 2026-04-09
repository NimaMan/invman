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
    evaluate_van_roy_case_study,
    get_exact_dp_summary,
    get_exact_verification_reference,
    get_van_roy_case_study,
    implied_target_cost_from_savings_pct,
    list_references,
    search_constant_base_stock,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Rust multi-echelon package against the repo exact verifier, the public Gijs settings, and the open Van Roy case-study row."
    )
    parser.add_argument("--gijs_replications", type=int, default=8)
    parser.add_argument("--gijs_horizon", type=int, default=20000)
    parser.add_argument("--van_roy_replications", type=int, default=8)
    parser.add_argument("--van_roy_horizon", type=int, default=20000)
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


def _gijs_validation(replications: int, horizon: int, seed: int) -> list[dict]:
    rows = []
    for reference in list_references():
        baseline = search_constant_base_stock(
            reference,
            allocation_mode=reference["policy_allocation_mode"],
            replications=replications,
            horizon=horizon,
            seed=seed,
        )
        best = dict(baseline["best_result"])
        a3c_savings = reference["published_a3c_savings_pct"]
        van_roy_savings = reference["published_van_roy_savings_pct_approx"]
        rows.append(
            {
                "reference": reference["name"],
                "literature_verified": bool(reference["literature_verified"]),
                "baseline": best,
                "published_a3c_savings_pct": a3c_savings,
                "published_van_roy_savings_pct_approx": van_roy_savings,
                "implied_a3c_target_cost": implied_target_cost_from_savings_pct(
                    float(best["mean_cost"]),
                    a3c_savings,
                ),
                "implied_van_roy_target_cost": implied_target_cost_from_savings_pct(
                    float(best["mean_cost"]),
                    van_roy_savings,
                ),
                "evaluation_protocol": {
                    "replications": int(replications),
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "allocation_mode": reference["policy_allocation_mode"],
                    "warehouse_base_stock_mode": reference["warehouse_base_stock_mode"],
                },
            }
        )
    return rows


def _van_roy_validation(replications: int, horizon: int, seed: int) -> dict:
    reference = get_van_roy_case_study()
    return {
        "reference": reference,
        "evaluations": {
            mode: evaluate_van_roy_case_study(
                reference,
                allocation_mode=mode,
                replications=replications,
                horizon=horizon,
                seed=seed,
            )
            for mode in ("sequential_index", "proportional", "min_shortage")
        },
    }


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
        "| Gijs Setting | Repo Constant Base-Stock Cost | Published A3C Savings % | Implied A3C Target Cost |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["gijs_validation"]:
        target = row["implied_a3c_target_cost"]
        lines.append(
            f"| `{row['reference']}` | `{row['baseline']['mean_cost']:.3f}` | `{row['published_a3c_savings_pct']}` | `{target:.3f}` |"
        )
    lines.extend(
        [
            "",
            "| Van Roy Allocation Mode | Repo Cost at Published (330,23) | Published Cost | Gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode, result in payload["van_roy_validation"]["evaluations"].items():
        lines.append(
            f"| `{mode}` | `{result['mean_cost']:.3f}` | `{result['published_constant_base_stock_mean_cost']:.3f}` | `{result['gap_vs_published_cost']:.3f}` |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation(),
        "gijs_validation": _gijs_validation(
            replications=parsed.gijs_replications,
            horizon=parsed.gijs_horizon,
            seed=parsed.seed,
        ),
        "van_roy_validation": _van_roy_validation(
            replications=parsed.van_roy_replications,
            horizon=parsed.van_roy_horizon,
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
