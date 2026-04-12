from __future__ import annotations

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

from common import dumps_json, ensure_parent, list_references  # noqa: E402

import invman_rust  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Van Roy literature protocol against the current Rust multi-echelon "
            "implementation by sweeping heuristic horizons and warm-up choices at the "
            "published base-stock levels."
        )
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[200, 1_000, 10_000, 100_000],
    )
    parser.add_argument(
        "--warm_up_ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.1],
    )
    parser.add_argument(
        "--allocation_modes",
        nargs="+",
        default=["min_shortage", "proportional"],
    )
    parser.add_argument("--replications", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _evaluate(reference: dict, allocation_mode: str, horizon: int, warm_up_ratio: float, replications: int, seed: int) -> dict:
    published_levels = [int(value) for value in reference["published_constant_base_stock_levels"]]
    result = dict(
        invman_rust.multi_echelon_search_stationary_policy(
            policy_kind="regular_base_stock",
            allocation_mode=str(allocation_mode),
            warehouse_levels=[published_levels[0]],
            retailer_levels=[published_levels[1]],
            warehouse_lead_time=int(reference["warehouse_lead_time"]),
            retailer_lead_time=int(reference["retailer_lead_time"]),
            num_retailers=int(reference["num_retailers"]),
            warehouse_holding_cost=float(reference["warehouse_holding_cost"]),
            retailer_holding_cost=float(reference["retailer_holding_cost"]),
            warehouse_expedited_cost=float(reference["warehouse_expedited_cost"]),
            warehouse_lost_sale_cost=float(reference["warehouse_lost_sale_cost"]),
            expedited_service_prob=float(reference["expedited_service_prob"]),
            warehouse_capacity=int(reference["warehouse_capacity"]),
            warehouse_inventory_cap=int(reference["warehouse_inventory_cap"]),
            retailer_inventory_cap=int(reference["retailer_inventory_cap"]),
            inventory_dynamics_mode=str(reference["inventory_dynamics_mode"]),
            demand_distribution=str(reference["demand_distribution"]),
            demand_mean=float(reference["demand_mean"]),
            demand_std=float(reference["demand_std"]),
            horizon=int(horizon),
            replications=int(replications),
            seed=int(seed),
            warm_up_periods_ratio=float(warm_up_ratio),
            discount_factor=1.0,
            objective="average_cost_after_warmup",
            top_k=1,
        )
    )
    best = dict(result["best_result"])
    published_cost = float(reference["published_constant_base_stock_mean_cost"])
    return {
        "reference": str(reference["name"]),
        "allocation_mode": str(allocation_mode),
        "published_levels": published_levels,
        "published_cost": published_cost,
        "horizon": int(horizon),
        "warm_up_ratio": float(warm_up_ratio),
        "replications": int(replications),
        "seed": int(seed),
        "repo_mean_cost": float(best["mean_cost"]),
        "repo_cost_std": float(best["cost_std"]),
        "gap_vs_published": float(best["mean_cost"] - published_cost),
    }


def main():
    parsed = parse_args()
    literature_references = [
        reference
        for reference in list_references()
        if reference.get("published_constant_base_stock_mean_cost") is not None
    ]

    rows = []
    for reference in literature_references:
        for allocation_mode in parsed.allocation_modes:
            for horizon in parsed.horizons:
                for warm_up_ratio in parsed.warm_up_ratios:
                    rows.append(
                        _evaluate(
                            reference,
                            allocation_mode=str(allocation_mode),
                            horizon=int(horizon),
                            warm_up_ratio=float(warm_up_ratio),
                            replications=int(parsed.replications),
                            seed=int(parsed.seed),
                        )
                    )

    payload = {
        "current_repo_protocol": {
            "initial_state_convention": (
                "zero_state; initialize_random_state currently returns zero on-hand inventory "
                "and zero pipeline for every replication"
            ),
            "evaluation_objective": "average_cost_after_warmup",
            "notes": [
                "This audit isolates horizon and warm-up sensitivity at the published base-stock levels.",
                "It does not change the underlying zero-state initialization currently used by the Rust runtime.",
            ],
        },
        "paper_protocol_questions": [
            "Van Roy states the heuristic rows were computed from a lengthy simulation.",
            "The paper does not state a single explicit heuristic warm-up ratio.",
            "The paper does not state a single explicit heuristic initial-state convention.",
            "The NDP figures average costs over rolling finite windows during one long simulation run, which is not the same as the heuristic exhaustive-search protocol.",
        ],
        "results": rows,
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))


if __name__ == "__main__":
    main()
