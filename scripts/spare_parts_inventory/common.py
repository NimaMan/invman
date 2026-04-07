from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policies.soft_tree import SoftTreePolicy

import invman_rust


def get_primary_reference() -> dict:
    return dict(invman_rust.spare_parts_inventory_primary_reference_instance())


def get_literature_benchmark_catalog() -> list[dict]:
    return [
        dict(entry)
        for entry in invman_rust.spare_parts_inventory_literature_benchmark_catalog()
    ]


def get_kranenburg_reference_instances() -> list[dict]:
    return [
        dict(entry)
        for entry in invman_rust.spare_parts_inventory_kranenburg_reference_instances()
    ]


def get_kranenburg_exact_summary(instance_name: str | None = None) -> dict:
    kwargs = {} if instance_name is None else {"instance_name": str(instance_name)}
    return dict(invman_rust.spare_parts_inventory_kranenburg_exact_summary(**kwargs))


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.spare_parts_inventory_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.spare_parts_inventory_exact_dp_summary())


def base_stock_params(reference: dict) -> list[int]:
    return [int(reference["benchmark_base_stock_level"])]


def lead_time_mean_cover_params(reference: dict) -> list[float]:
    return [float(reference["benchmark_lead_time_mean_cover_safety_buffer"])]


def default_action_cap(reference: dict) -> int:
    mean_cover_target = int(
        invman_rust.spare_parts_inventory_lead_time_mean_cover_target(
            installed_base=int(reference["installed_base"]),
            failure_probability=float(reference["failure_probability"]),
            procurement_lead_time=int(reference["procurement_lead_time"]),
            safety_buffer=float(reference["benchmark_lead_time_mean_cover_safety_buffer"]),
        )
    )
    return max(
        16,
        int(reference["installed_base"]),
        2 * int(reference["benchmark_base_stock_level"]),
        2 * mean_cover_target,
    )


def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    *,
    replications: int,
    seed: int,
    params: list[float] | None = None,
) -> dict:
    if params is None:
        if policy_name == "base_stock":
            params = base_stock_params(reference)
        elif policy_name == "lead_time_mean_cover":
            params = lead_time_mean_cover_params(reference)
        else:
            raise ValueError(f"unknown heuristic policy '{policy_name}'")
    summary = dict(
        invman_rust.spare_parts_inventory_simulate_policy(
            policy_name=str(policy_name),
            params=[float(value) for value in params],
            on_hand_inventory=int(reference["initial_on_hand_inventory"]),
            backlog=int(reference["initial_backlog"]),
            procurement_pipeline=[
                int(value) for value in reference["initial_procurement_pipeline"]
            ],
            repair_pipeline=[int(value) for value in reference["initial_repair_pipeline"]],
            installed_base=int(reference["installed_base"]),
            periods=int(reference["periods"]),
            failure_probability=float(reference["failure_probability"]),
            holding_cost=float(reference["holding_cost"]),
            downtime_cost=float(reference["downtime_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            replications=int(replications),
            seed=int(seed),
            discount_factor=0.99,
        )
    )
    summary["params"] = [float(value) for value in params]
    summary["num_samples"] = int(replications)
    return summary


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    action_cap: int | None = None,
) -> SoftTreePolicy:
    input_dim = (
        len(reference["initial_procurement_pipeline"])
        + len(reference["initial_repair_pipeline"])
        + 7
    )
    return SoftTreePolicy(
        input_dim=int(input_dim),
        action_spec={
            "action_dim": 1,
            "action_mode": "scalar_quantity",
            "min_values": [0],
            "max_values": [default_action_cap(reference) if action_cap is None else int(action_cap)],
            "allowed_values": None,
        },
        depth=int(depth),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        state_normalizer="identity",
        state_scale=None,
    )


def soft_tree_rollout_kwargs(reference: dict, model: SoftTreePolicy, *, flat_params) -> dict:
    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.action_spec["min_values"]],
        "max_values": [int(value) for value in model.action_spec["max_values"]],
        "action_mode": str(model.action_spec["action_mode"]),
        "on_hand_inventory": int(reference["initial_on_hand_inventory"]),
        "backlog": int(reference["initial_backlog"]),
        "procurement_pipeline": [
            int(value) for value in reference["initial_procurement_pipeline"]
        ],
        "repair_pipeline": [int(value) for value in reference["initial_repair_pipeline"]],
        "installed_base": int(reference["installed_base"]),
        "periods": int(reference["periods"]),
        "failure_probability": float(reference["failure_probability"]),
        "holding_cost": float(reference["holding_cost"]),
        "downtime_cost": float(reference["downtime_cost"]),
        "procurement_cost": float(reference["procurement_cost"]),
        "discount_factor": 0.99,
        "temperature": float(model.temperature),
        "split_type": str(model.split_type),
        "leaf_type": str(model.leaf_type),
        "allowed_values": model.action_spec.get("allowed_values"),
    }


def evaluate_soft_tree_policy(
    reference: dict,
    model: SoftTreePolicy,
    seeds: Iterable[int],
    *,
    flat_params=None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    costs = []
    for seed in seeds:
        discounted_cost = invman_rust.spare_parts_inventory_soft_tree_rollout(
            seed=int(seed),
            **soft_tree_rollout_kwargs(reference, model, flat_params=params),
        )
        costs.append(float(discounted_cost))
    costs = np.asarray(costs, dtype=np.float64)
    return {
        "mean_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_samples": int(costs.size),
    }


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
