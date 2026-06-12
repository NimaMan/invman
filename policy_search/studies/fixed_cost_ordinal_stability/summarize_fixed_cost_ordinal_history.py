#!/usr/bin/env python3
"""Summarize stored fixed-cost ordinal-policy benchmark results."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO_ROOT / "outputs" / "benchmarks"
POLICIES = {
    "linear_gated_ordinal_quantity",
    "linear_soft_gated_ordinal_quantity",
    "nn_gated_ordinal_quantity",
    "nn_soft_gated_ordinal_quantity",
}


def infer_policy_name(payload: dict[str, object], result_path: Path) -> str | None:
    policy_name = payload.get("policy_name")
    if isinstance(policy_name, str) and policy_name in POLICIES:
        return policy_name

    candidates = [payload.get("experiment_name"), result_path.name]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        if "linear_gated_ordinal_quantity" in candidate:
            return "linear_gated_ordinal_quantity"
        if "linear_soft_gated_ordinal_quantity" in candidate:
            return "linear_soft_gated_ordinal_quantity"
        if "nn_gated_ordinal_quantity" in candidate:
            return "nn_gated_ordinal_quantity"
        if "nn_soft_gated_ordinal_quantity" in candidate:
            return "nn_soft_gated_ordinal_quantity"
    return None


def iter_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_path in BENCHMARK_ROOT.glob("**/results/*.json"):
        if result_path.name.startswith("status_"):
            continue
        try:
            payload = json.loads(result_path.read_text())
        except Exception:
            continue
        if payload.get("problem") != "lost_sales_fixed_order_cost":
            continue
        policy_name = infer_policy_name(payload, result_path)
        if policy_name is None:
            continue
        learned = ((payload.get("evaluation") or {}).get("learned_policy") or {})
        mean_cost = learned.get("mean_cost")
        if mean_cost is None:
            continue
        problem_params = payload.get("problem_params") or {}
        rows.append(
            {
                "result_path": result_path,
                "experiment_name": payload.get("experiment_name"),
                "policy_name": policy_name,
                "rollout_backend": payload.get("rollout_backend"),
                "training_episodes": payload.get("training_episodes"),
                "training_horizon": payload.get("training_horizon"),
                "evaluation_horizon": payload.get("evaluation_horizon"),
                "max_order_size": payload.get("max_order_size"),
                "lead_time": problem_params.get("lead_time"),
                "shortage_cost": payload.get("shortage_cost")
                or problem_params.get("shortage_cost"),
                "fixed_order_cost": payload.get("fixed_order_cost")
                or problem_params.get("fixed_order_cost"),
                "demand_dist_name": payload.get("demand_dist_name"),
                "demand_rate": payload.get("demand_rate"),
                "mean_cost": mean_cost,
                "std_cost": learned.get("std_cost"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["fixed_order_cost"],
            row["shortage_cost"],
            row["lead_time"],
            row["training_episodes"],
            row["policy_name"],
            row["mean_cost"],
        )
    )
    return rows


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    rows = iter_rows()
    headers = [
        "policy",
        "backend",
        "L",
        "p",
        "K",
        "train_eps",
        "Q",
        "mean_cost",
        "std_cost",
        "experiment_name",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["policy_name"],
                row["rollout_backend"],
                row["lead_time"],
                row["shortage_cost"],
                row["fixed_order_cost"],
                row["training_episodes"],
                row["max_order_size"],
                row["mean_cost"],
                row["std_cost"],
                row["experiment_name"],
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(format_value(value)))

    def render_row(values: list[object]) -> str:
        return "  ".join(
            format_value(value).ljust(widths[idx]) for idx, value in enumerate(values)
        )

    print(render_row(headers))
    print(render_row(["-" * width for width in widths]))
    for row in table_rows:
        print(render_row(row))


if __name__ == "__main__":
    main()
