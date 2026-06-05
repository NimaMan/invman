from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policy import Policy

import invman_rust


def get_primary_reference() -> dict:
    return dict(invman_rust.procurement_removal_inventory_primary_reference_instance())


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.procurement_removal_inventory_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.procurement_removal_inventory_exact_dp_summary())


def interval_stock_params(reference: dict) -> list[int]:
    return [
        int(reference["benchmark_order_up_to"]),
        int(reference["benchmark_remove_down_to"]),
    ]


def returnability_buffer_params(reference: dict) -> list[int]:
    return [
        int(reference["benchmark_order_up_to"]),
        int(reference["benchmark_remove_down_to"]),
        int(reference["benchmark_returnable_buffer"]),
    ]


def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    seeds: Iterable[int],
    *,
    params: list[int] | None = None,
) -> dict:
    if params is None:
        if policy_name == "interval_stock":
            params = interval_stock_params(reference)
        elif policy_name == "returnability_buffer_interval_stock":
            params = returnability_buffer_params(reference)
        else:
            raise ValueError(f"unknown heuristic policy '{policy_name}'")
    return dict(
        invman_rust.procurement_removal_inventory_simulate_policy(
            policy_name=policy_name,
            params=[int(value) for value in params],
            inventory_level=int(reference["initial_inventory_level"]),
            returnable_inventory=int(reference["initial_returnable_inventory"]),
            periods=int(reference["periods"]),
            seeds=[int(seed) for seed in seeds],
            demand_kind=str(reference["demand_distribution_kind"]),
            demand_mean=float(reference["demand_mean"]),
            returnable_purchase_cap=int(reference["returnable_purchase_cap"]),
            purchase_cost_per_unit=float(reference["purchase_cost_per_unit"]),
            return_value_per_unit=float(reference["return_value_per_unit"]),
            liquidation_value_per_unit=float(reference["liquidation_value_per_unit"]),
            holding_cost_per_unit=float(reference["holding_cost_per_unit"]),
            shortage_cost_per_unit=float(reference["shortage_cost_per_unit"]),
            max_purchase_quantity=int(reference["max_purchase_quantity"]),
            max_removal_quantity=int(reference["max_removal_quantity"]),
            discount_factor=0.99,
        )
    )


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
):
    max_purchase_quantity = int(reference["max_purchase_quantity"])
    max_removal_quantity = int(reference["max_removal_quantity"])
    return Policy(
        backbone="soft_tree",
        input_dim=7,
        control_dim=2,
        control_mode="vector_quantity",
        min_values=(0, 0),
        max_values=(max_purchase_quantity, max_removal_quantity),
        allowed_values=None,
        max_order_size=max(max_purchase_quantity, max_removal_quantity),
        depth=int(depth),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        state_normalizer="identity",
        state_scale=None,
    )


def soft_tree_rollout_kwargs(
    reference: dict,
    model: Policy,
    *,
    flat_params,
) -> dict:
    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.min_values],
        "max_values": [int(value) for value in model.max_values],
        "action_mode": str(model.control_mode),
        "inventory_level": int(reference["initial_inventory_level"]),
        "returnable_inventory": int(reference["initial_returnable_inventory"]),
        "periods": int(reference["periods"]),
        "demand_kind": str(reference["demand_distribution_kind"]),
        "demand_mean": float(reference["demand_mean"]),
        "returnable_purchase_cap": int(reference["returnable_purchase_cap"]),
        "purchase_cost_per_unit": float(reference["purchase_cost_per_unit"]),
        "return_value_per_unit": float(reference["return_value_per_unit"]),
        "liquidation_value_per_unit": float(reference["liquidation_value_per_unit"]),
        "holding_cost_per_unit": float(reference["holding_cost_per_unit"]),
        "shortage_cost_per_unit": float(reference["shortage_cost_per_unit"]),
        "max_purchase_quantity": int(reference["max_purchase_quantity"]),
        "max_removal_quantity": int(reference["max_removal_quantity"]),
        "discount_factor": 0.99,
        "temperature": float(model.temperature),
        "split_type": str(model.split_type),
        "leaf_type": str(model.leaf_type),
        "allowed_values": model.allowed_values,
    }


def evaluate_soft_tree_policy(
    reference: dict,
    model: Policy,
    seeds: Iterable[int],
    *,
    flat_params=None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    costs = []
    for seed in seeds:
        discounted_cost = invman_rust.procurement_removal_inventory_soft_tree_rollout(
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
