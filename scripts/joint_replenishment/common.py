from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policies.soft_tree import SoftTreePolicy

import invman_rust


def get_reference(name: str) -> dict:
    return dict(invman_rust.joint_replenishment_get_reference_instance(str(name)))


def list_references() -> list[dict]:
    return [dict(reference) for reference in invman_rust.joint_replenishment_list_reference_instances()]


def get_primary_reference() -> dict:
    return dict(invman_rust.joint_replenishment_primary_reference_instance())


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.joint_replenishment_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.joint_replenishment_exact_dp_summary())


def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    params: list[float],
    *,
    replications: int,
    seed: int,
) -> dict:
    mean_cost, cost_std = invman_rust.joint_replenishment_simulate_policy(
        policy_name=str(policy_name),
        params=[float(value) for value in params],
        initial_inventory_levels=[int(value) for value in reference["initial_inventory_levels"]],
        periods=int(reference["periods"]),
        replications=int(replications),
        seed=int(seed),
        demand_lows=[int(value) for value in reference["demand_lows"]],
        demand_highs=[int(value) for value in reference["demand_highs"]],
        truck_capacity=int(reference["truck_capacity"]),
        minor_order_costs=[float(value) for value in reference["minor_order_costs"]],
        major_order_cost=float(reference["major_order_cost"]),
        holding_costs=[float(value) for value in reference["holding_costs"]],
        shortage_costs=[float(value) for value in reference["shortage_costs"]],
        discount_factor=float(reference["discount_factor"]),
    )
    return {
        "mean_cost": float(mean_cost),
        "cost_std": float(cost_std),
        "num_samples": int(replications),
        "params": [float(value) for value in params],
    }


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
) -> SoftTreePolicy:
    return SoftTreePolicy(
        input_dim=4,
        action_spec={
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [int(reference["max_order_quantities"][0]), int(reference["max_order_quantities"][1])],
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
        "initial_inventory_levels": [int(value) for value in reference["initial_inventory_levels"]],
        "demand_lows": [int(value) for value in reference["demand_lows"]],
        "demand_highs": [int(value) for value in reference["demand_highs"]],
        "truck_capacity": int(reference["truck_capacity"]),
        "minor_order_costs": [float(value) for value in reference["minor_order_costs"]],
        "major_order_cost": float(reference["major_order_cost"]),
        "holding_costs": [float(value) for value in reference["holding_costs"]],
        "shortage_costs": [float(value) for value in reference["shortage_costs"]],
        "periods": int(reference["periods"]),
        "discount_factor": float(reference["discount_factor"]),
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
        discounted_cost = invman_rust.joint_replenishment_soft_tree_rollout(
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
