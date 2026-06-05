from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policy import Policy

import invman_rust


def get_primary_reference() -> dict:
    return dict(invman_rust.joint_pricing_inventory_primary_reference_instance())


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.joint_pricing_inventory_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.joint_pricing_inventory_exact_dp_summary())


def static_price_base_stock_params(reference: dict) -> list[int]:
    return [
        int(reference["benchmark_static_order_up_to"]),
        int(reference["benchmark_static_price_index"]),
    ]


def inventory_sensitive_base_stock_params(reference: dict) -> list[int]:
    return [
        int(reference["benchmark_inventory_sensitive_order_up_to"]),
        int(reference["benchmark_markdown_threshold"]),
        int(reference["benchmark_high_price_index"]),
        int(reference["benchmark_low_price_index"]),
    ]


def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    *,
    replications: int,
    seed: int,
    params: list[float] | None = None,
) -> dict:
    if params is None:
        if policy_name == "static_price_base_stock":
            params = static_price_base_stock_params(reference)
        elif policy_name == "inventory_sensitive_base_stock":
            params = inventory_sensitive_base_stock_params(reference)
        else:
            raise ValueError(f"unknown heuristic policy '{policy_name}'")
    summary = dict(
        invman_rust.joint_pricing_inventory_simulate_policy(
        policy_name=str(policy_name),
        params=[float(value) for value in params],
        inventory_level=int(reference["initial_inventory_level"]),
        periods=int(reference["periods"]),
        replications=int(replications),
        seed=int(seed),
        demand_kind=str(reference["demand_distribution_kind"]),
        price_levels=[float(value) for value in reference["price_levels"]],
        demand_means=[float(value) for value in reference["price_demand_means"]],
        procurement_cost_per_unit=float(reference["procurement_cost_per_unit"]),
        holding_cost_per_unit=float(reference["holding_cost_per_unit"]),
        stockout_cost_per_unit=float(reference["stockout_cost_per_unit"]),
        max_order_quantity=int(reference["max_order_quantity"]),
        discount_factor=0.99,
        salvage_value_per_unit=float(reference["salvage_value_per_unit"]),
    )
    )
    return {
        "mean_cost": float(summary["mean_discounted_cost"]),
        "cost_std": float(summary["std_discounted_cost"]),
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
) -> Policy:
    max_order_quantity = int(reference["max_order_quantity"])
    return Policy(
        backbone="soft_tree",
        input_dim=7,
        control_dim=2,
        control_mode="vector_quantity",
        min_values=(0, 0),
        max_values=(max_order_quantity, len(reference["price_levels"]) - 1),
        allowed_values=None,
        max_order_size=max_order_quantity,
        depth=int(depth),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        state_normalizer="identity",
        state_scale=None,
    )


def soft_tree_rollout_kwargs(reference: dict, model: Policy, *, flat_params) -> dict:
    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.min_values],
        "max_values": [int(value) for value in model.max_values],
        "action_mode": str(model.control_mode),
        "inventory_level": int(reference["initial_inventory_level"]),
        "periods": int(reference["periods"]),
        "demand_kind": str(reference["demand_distribution_kind"]),
        "price_levels": [float(value) for value in reference["price_levels"]],
        "demand_means": [float(value) for value in reference["price_demand_means"]],
        "procurement_cost_per_unit": float(reference["procurement_cost_per_unit"]),
        "holding_cost_per_unit": float(reference["holding_cost_per_unit"]),
        "stockout_cost_per_unit": float(reference["stockout_cost_per_unit"]),
        "salvage_value_per_unit": float(reference["salvage_value_per_unit"]),
        "max_order_quantity": int(reference["max_order_quantity"]),
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
        discounted_cost = invman_rust.joint_pricing_inventory_soft_tree_rollout(
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
