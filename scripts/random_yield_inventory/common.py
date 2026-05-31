from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

# NOTE (2026-05): the soft-tree descriptor moved from the removed `invman.policies.soft_tree`
# module to `invman.policy.Policy(backbone="soft_tree", ...)`. The exact-DP / heuristic validation
# and literature-summary scripts in this folder do NOT need a soft-tree class, so this import is made
# lazy: importing common.py no longer fails, and the soft-tree helpers below resolve the class only
# if/when they are actually called. New work should prefer
# scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py, which uses the
# current Policy interface directly.
try:  # pragma: no cover - legacy path, kept for backward compatibility only
    from invman.policies.soft_tree import SoftTreePolicy  # type: ignore
except Exception:  # ModuleNotFoundError on current package layout
    SoftTreePolicy = None  # type: ignore

import invman_rust


def get_primary_reference() -> dict:
    return dict(invman_rust.random_yield_inventory_primary_reference_instance())


def get_literature_benchmark_families() -> list[dict]:
    return [
        dict(family)
        for family in invman_rust.random_yield_inventory_literature_benchmark_families()
    ]


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.random_yield_inventory_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.random_yield_inventory_exact_dp_summary())


def linear_inflation_params(reference: dict) -> list[float]:
    target_stock_level, yield_inflation_factor = invman_rust.random_yield_inventory_linear_inflation_parameters(
        demand_mean=float(reference["demand_mean"]),
        success_probability=float(reference["success_probability"]),
        lead_time=int(reference["lead_time"]),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
    )
    return [float(target_stock_level), float(yield_inflation_factor)]


def default_action_cap(reference: dict) -> int:
    lead_time = int(reference["lead_time"])
    demand_mean = float(reference["demand_mean"])
    target_stock_level, yield_inflation_factor = linear_inflation_params(reference)
    demand_cover = 4.0 * demand_mean * (lead_time + 1)
    inflated_target = yield_inflation_factor * target_stock_level
    return int(math.ceil(max(32.0, demand_cover, inflated_target)))


def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    seeds: Iterable[int],
    *,
    params: list[float] | None = None,
    demand_distribution: str = "poisson",
) -> dict:
    if params is None:
        if policy_name == "linear_inflation":
            params = linear_inflation_params(reference)
        else:
            params = []
    return dict(
        invman_rust.random_yield_inventory_policy_discounted_cost_summary(
            policy_name=policy_name,
            params=[float(value) for value in params],
            initial_inventory_level=float(reference["initial_inventory_level"]),
            pipeline_orders=[float(value) for value in reference["initial_pipeline_orders"]],
            periods=int(reference["periods"]),
            seeds=[int(seed) for seed in seeds],
            demand_mean=float(reference["demand_mean"]),
            success_probability=float(reference["success_probability"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            discount_factor=float(reference["discount_factor"]),
            demand_distribution=demand_distribution,
        )
    )


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    action_cap: int | None = None,
) -> SoftTreePolicy:
    return SoftTreePolicy(
        input_dim=int(reference["lead_time"]) + 3,
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


def soft_tree_rollout_kwargs(
    reference: dict,
    model: SoftTreePolicy,
    *,
    flat_params,
) -> dict:
    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.action_spec["min_values"]],
        "max_values": [int(value) for value in model.action_spec["max_values"]],
        "action_mode": str(model.action_spec["action_mode"]),
        "initial_inventory_level": float(reference["initial_inventory_level"]),
        "pipeline_orders": [float(value) for value in reference["initial_pipeline_orders"]],
        "periods": int(reference["periods"]),
        "demand_mean": float(reference["demand_mean"]),
        "success_probability": float(reference["success_probability"]),
        "holding_cost": float(reference["holding_cost"]),
        "shortage_cost": float(reference["shortage_cost"]),
        "procurement_cost": float(reference["procurement_cost"]),
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
    demand_distribution: str = "poisson",
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    costs = []
    for seed in seeds:
        discounted_cost = invman_rust.random_yield_inventory_soft_tree_rollout(
            seed=int(seed),
            demand_distribution=demand_distribution,
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
