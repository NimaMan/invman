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
    MEDIUM_REFERENCE_INSTANCE_NAME,
    dumps_json,
    ensure_parent,
    evaluate_heuristic_policy,
    get_reference,
    search_best_base_stock,
    search_best_bsp_low_ew,
)

import invman_rust


EXACT_VERIFICATION_REFERENCES = (
    "de_moor2022_m2_exp1_l1_cp7_lifo",
    "de_moor2022_m2_exp2_l1_cp7_fifo",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the Rust perishable-inventory implementation against paper benchmarks.")
    parser.add_argument("--simulation_reference", default=MEDIUM_REFERENCE_INSTANCE_NAME)
    parser.add_argument("--simulation_seeds", type=int, default=128)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _exact_validation_rows():
    rows = []
    for reference_name in EXACT_VERIFICATION_REFERENCES:
        reference = get_reference(reference_name)
        exact = dict(invman_rust.perishable_inventory_exact_mdp_summary(reference_name))
        figure = reference.get("published_figure3_verification")
        rows.append(
            {
                "reference_instance": reference_name,
                "published_value_iteration_mean_return": reference["published_scenario_a_returns"]["value_iteration_mean_return"],
                "repo_value_iteration_mean_return_rounded": exact["value_iteration_mean_return_rounded"],
                "published_base_stock_level": None if figure is None else figure["published_base_stock_level"],
                "repo_best_base_stock_level": exact["best_base_stock_level"],
                "policy_table_match": bool(exact.get("matches_published_policy_table", False)),
            }
        )
    return rows


def _simulation_validation(reference_name: str, num_seeds: int):
    reference = get_reference(reference_name)
    seeds = list(range(123, 123 + num_seeds))
    base_stock_search = search_best_base_stock(reference, seeds)
    bsp_low_ew_search = search_best_bsp_low_ew(reference, seeds)
    best_base_stock_eval = evaluate_heuristic_policy(
        reference,
        "base_stock",
        tuple(base_stock_search["best"]["params"]),
        seeds,
    )
    best_bsp_low_ew_eval = evaluate_heuristic_policy(
        reference,
        "bsp_low_ew",
        tuple(bsp_low_ew_search["best"]["params"]),
        seeds,
    )
    published = reference["published_scenario_a_returns"]
    return {
        "reference_instance": reference_name,
        "published_value_iteration_mean_return": published["value_iteration_mean_return"],
        "published_best_base_stock_mean_return": published["best_base_stock_mean_return"],
        "repo_base_stock": best_base_stock_eval,
        "repo_bsp_low_ew": best_bsp_low_ew_eval,
        "repo_base_stock_return_gap_vs_paper": (
            best_base_stock_eval["mean_return"] - published["best_base_stock_mean_return"]
        ),
        "repo_bsp_low_ew_return_gap_vs_paper_vi": (
            best_bsp_low_ew_eval["mean_return"] - published["value_iteration_mean_return"]
        ),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "| Instance | Published VI Return | Repo Exact VI Return | Published BS Level | Repo BS Level | Policy Table Match |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["exact_validation"]:
        lines.append(
            f"| `{row['reference_instance']}` | `{row['published_value_iteration_mean_return']}` | "
            f"`{row['repo_value_iteration_mean_return_rounded']}` | "
            f"`{row['published_base_stock_level']}` | `{row['repo_best_base_stock_level']}` | "
            f"`{row['policy_table_match']}` |"
        )

    simulation = payload["simulation_validation"]
    lines.extend(
        [
            "",
            "| Policy | Params | Mean Return | Gap vs Published Paper Value |",
            "| --- | --- | ---: | ---: |",
            f"| `published_value_iteration` | `-` | `{simulation['published_value_iteration_mean_return']:.3f}` | `0.000` |",
            f"| `published_base_stock` | `-` | `{simulation['published_best_base_stock_mean_return']:.3f}` | `0.000` |",
            f"| `repo_base_stock` | `{simulation['repo_base_stock']['params']}` | "
            f"`{simulation['repo_base_stock']['mean_return']:.3f}` | "
            f"`{simulation['repo_base_stock_return_gap_vs_paper']:.3f}` |",
            f"| `repo_bsp_low_ew` | `{simulation['repo_bsp_low_ew']['params']}` | "
            f"`{simulation['repo_bsp_low_ew']['mean_return']:.3f}` | "
            f"`{simulation['repo_bsp_low_ew_return_gap_vs_paper_vi']:.3f}` |",
        ]
    )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    payload = {
        "exact_validation": _exact_validation_rows(),
        "simulation_validation": _simulation_validation(
            parsed.simulation_reference,
            parsed.simulation_seeds,
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
