from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np

from invman.policies.soft_tree import SoftTreePolicy

import invman_rust


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 1e-8:
        return -20.0
    if value > 20.0:
        return value
    return float(value + np.log(-np.expm1(-value)))


def initialize_soft_tree_to_constant_action(model: SoftTreePolicy, action: Iterable[int]) -> SoftTreePolicy:
    action_values = [int(value) for value in action]
    if len(action_values) != int(model.control_dim):
        raise ValueError(
            f"action length {len(action_values)} does not match model control_dim {model.control_dim}"
        )

    model.split_weights.fill(0.0)
    model.split_bias.fill(0.0)

    min_values = [int(value) for value in model.min_values]
    max_values = [int(value) for value in model.max_values]

    if model.leaf_type == "constant":
        for leaf_idx in range(int(model.num_leaves)):
            for dim_idx, target in enumerate(action_values):
                low = float(min_values[dim_idx])
                high = float(max_values[dim_idx])
                span = max(high - low, 1e-6)
                scaled = min(max((float(target) - low) / span, 1e-6), 1.0 - 1e-6)
                model.leaf_logits[leaf_idx, dim_idx] = float(np.log(scaled / (1.0 - scaled)))
        return model

    if model.leaf_type in {"linear", "sigmoid_linear"}:
        model.leaf_weights.fill(0.0)
        for leaf_idx in range(int(model.num_leaves)):
            for dim_idx, target in enumerate(action_values):
                low = float(min_values[dim_idx])
                high = float(max_values[dim_idx])
                if model.leaf_type == "linear":
                    model.leaf_bias[leaf_idx, dim_idx] = _inverse_softplus(float(target) - low)
                else:
                    span = max(high - low, 1e-6)
                    scaled = min(max((float(target) - low) / span, 1e-6), 1.0 - 1e-6)
                    model.leaf_bias[leaf_idx, dim_idx] = float(np.log(scaled / (1.0 - scaled)))
        return model

    raise NotImplementedError(f"unsupported soft-tree leaf type '{model.leaf_type}'")


def get_benchmark_reference() -> dict:
    return dict(invman_rust.multi_echelon_benchmark_reference())


def list_references() -> list[dict]:
    return [dict(reference) for reference in invman_rust.multi_echelon_list_reference_instances()]


def get_reference(name: str) -> dict:
    return dict(invman_rust.multi_echelon_get_reference_instance(str(name)))


def get_primary_reference() -> dict:
    return dict(invman_rust.multi_echelon_primary_reference_instance())


def get_van_roy_case_study() -> dict:
    return dict(invman_rust.multi_echelon_van_roy_case_study())


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.multi_echelon_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.multi_echelon_exact_dp_summary())


def is_exact_reference(reference: dict) -> bool:
    return "action_warehouse_levels" in reference


def benchmark_periods(reference: dict) -> int:
    return int(reference["periods"]) if is_exact_reference(reference) else int(reference["benchmark_periods"])


def benchmark_replications(reference: dict) -> int:
    return 1 if is_exact_reference(reference) else int(reference["benchmark_replications"])


def benchmark_warm_up_periods_ratio(reference: dict) -> float:
    return 0.0 if is_exact_reference(reference) else float(reference["warm_up_periods_ratio"])


def benchmark_rollout_objective(reference: dict) -> str:
    return "discounted_cost" if is_exact_reference(reference) else str(reference["rollout_objective"])


def benchmark_warehouse_base_stock_mode(reference: dict) -> str:
    return str(reference["warehouse_base_stock_mode"])


def benchmark_allocation_mode(reference: dict) -> str:
    return str(reference["allocation_mode"]) if is_exact_reference(reference) else str(reference["policy_allocation_mode"])


def action_grids(reference: dict) -> tuple[list[int], list[int]]:
    if is_exact_reference(reference):
        return (
            [int(value) for value in reference["action_warehouse_levels"]],
            [int(value) for value in reference["action_retailer_levels"]],
        )
    return (
        [int(value) for value in reference["benchmark_warehouse_levels"]],
        [int(value) for value in reference["benchmark_retailer_levels"]],
    )


def input_dim(
    reference: dict,
    *,
    include_period_feature: bool,
    policy_feature_mode: str = "full_decision_state",
) -> int:
    if policy_feature_mode == "full_decision_state":
        if str(reference.get("inventory_dynamics_mode", "gijs_2022")) == "van_roy_1997":
            base_dim = (
                1
                + int(reference["warehouse_lead_time"])
                + int(reference["num_retailers"]) * (1 + int(reference["retailer_lead_time"]))
            )
        else:
            base_dim = int(reference["warehouse_lead_time"]) + int(reference["num_retailers"]) * int(reference["retailer_lead_time"])
    elif policy_feature_mode == "symmetric_summary":
        base_dim = int(reference["warehouse_lead_time"]) + int(reference["retailer_lead_time"]) + 8
    elif policy_feature_mode == "compact_summary":
        base_dim = 22
    else:
        raise ValueError(f"unsupported policy_feature_mode '{policy_feature_mode}'")
    return int(base_dim + int(include_period_feature))


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    include_period_feature: bool | None = None,
    exact: bool | None = None,
    warehouse_levels: list[int] | None = None,
    retailer_levels: list[int] | None = None,
    policy_feature_mode: str = "full_decision_state",
) -> SoftTreePolicy:
    del exact
    if include_period_feature is None:
        include_period_feature = is_exact_reference(reference)
    if warehouse_levels is None or retailer_levels is None:
        warehouse_levels, retailer_levels = action_grids(reference)
    return SoftTreePolicy(
        input_dim=input_dim(
            reference,
            include_period_feature=bool(include_period_feature),
            policy_feature_mode=str(policy_feature_mode),
        ),
        action_spec={
            "action_dim": 2,
            "action_mode": "discrete_grid",
            "min_values": [int(warehouse_levels[0]), int(retailer_levels[0])],
            "max_values": [int(warehouse_levels[-1]), int(retailer_levels[-1])],
            "allowed_values": [warehouse_levels, retailer_levels],
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
    include_period_feature: bool | None = None,
    allocation_mode: str | None = None,
    warehouse_base_stock_mode: str | None = None,
    policy_feature_mode: str = "full_decision_state",
    policy_action_mode: str = "direct_base_stock",
    warehouse_anchor_level: int = 0,
    retailer_anchor_level: int = 0,
    reference_warehouse_levels: list[int] | None = None,
    reference_retailer_levels: list[int] | None = None,
) -> dict:
    if include_period_feature is None:
        include_period_feature = is_exact_reference(reference)
    if allocation_mode is None:
        allocation_mode = benchmark_allocation_mode(reference)
    if warehouse_base_stock_mode is None:
        warehouse_base_stock_mode = benchmark_warehouse_base_stock_mode(reference)

    common = {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.action_spec["min_values"]],
        "max_values": [int(value) for value in model.action_spec["max_values"]],
        "action_mode": str(model.action_spec["action_mode"]),
        "include_period_feature": bool(include_period_feature),
        "warehouse_base_stock_mode": str(warehouse_base_stock_mode),
        "allocation_mode": str(allocation_mode),
        "temperature": float(model.temperature),
        "split_type": str(model.split_type),
        "leaf_type": str(model.leaf_type),
        "allowed_values": model.action_spec.get("allowed_values"),
    }
    if is_exact_reference(reference):
        return {
            **common,
            "initial_warehouse_inventory": int(reference["initial_warehouse_inventory"]),
            "initial_warehouse_pipeline": [int(value) for value in reference["initial_warehouse_pipeline"]],
            "initial_retailer_inventory": [int(value) for value in reference["initial_retailer_inventory"]],
            "initial_retailer_pipeline": [
                [int(value) for value in row] for row in reference["initial_retailer_pipeline"]
            ],
            "demand_support": [int(value) for value in reference["demand_support"]],
            "demand_probabilities": [float(value) for value in reference["demand_probabilities"]],
            "periods": int(reference["periods"]),
            "discount_factor": float(reference["discount_factor"]),
            "warehouse_capacity": int(reference["warehouse_capacity"]),
            "warehouse_inventory_cap": int(reference["warehouse_inventory_cap"]),
            "retailer_inventory_cap": int(reference["retailer_inventory_cap"]),
            "inventory_dynamics_mode": str(reference["inventory_dynamics_mode"]),
            "warehouse_holding_cost": float(reference["warehouse_holding_cost"]),
            "retailer_holding_cost": float(reference["retailer_holding_cost"]),
            "warehouse_expedited_cost": float(reference["warehouse_expedited_cost"]),
            "warehouse_lost_sale_cost": float(reference["warehouse_lost_sale_cost"]),
            "expedited_service_prob": float(reference["expedited_service_prob"]),
        }

    return {
        **common,
        "policy_feature_mode": str(policy_feature_mode),
        "policy_action_mode": str(policy_action_mode),
        "warehouse_anchor_level": int(warehouse_anchor_level),
        "retailer_anchor_level": int(retailer_anchor_level),
        "reference_warehouse_levels": None
        if reference_warehouse_levels is None
        else [int(value) for value in reference_warehouse_levels],
        "reference_retailer_levels": None
        if reference_retailer_levels is None
        else [int(value) for value in reference_retailer_levels],
        "warehouse_lead_time": int(reference["warehouse_lead_time"]),
        "retailer_lead_time": int(reference["retailer_lead_time"]),
        "num_retailers": int(reference["num_retailers"]),
        "warehouse_holding_cost": float(reference["warehouse_holding_cost"]),
        "retailer_holding_cost": float(reference["retailer_holding_cost"]),
        "warehouse_expedited_cost": float(reference["warehouse_expedited_cost"]),
        "warehouse_lost_sale_cost": float(reference["warehouse_lost_sale_cost"]),
        "expedited_service_prob": float(reference["expedited_service_prob"]),
        "warehouse_capacity": int(reference["warehouse_capacity"]),
        "warehouse_inventory_cap": int(reference["warehouse_inventory_cap"]),
        "retailer_inventory_cap": int(reference["retailer_inventory_cap"]),
        "inventory_dynamics_mode": str(reference["inventory_dynamics_mode"]),
        "demand_distribution": str(reference["demand_distribution"]),
        "demand_mean": float(reference["demand_mean"]),
        "demand_std": float(reference["demand_std"]),
        "horizon": int(reference["benchmark_periods"]),
        "warm_up_periods_ratio": float(reference["warm_up_periods_ratio"]),
        "discount_factor": 1.0,
        "objective": str(reference["rollout_objective"]),
    }


def evaluate_soft_tree_policy(
    reference: dict,
    model: SoftTreePolicy,
    seeds: Iterable[int],
    *,
    flat_params=None,
    include_period_feature: bool | None = None,
    allocation_mode: str | None = None,
    warehouse_base_stock_mode: str | None = None,
    policy_feature_mode: str = "full_decision_state",
    policy_action_mode: str = "direct_base_stock",
    warehouse_anchor_level: int = 0,
    retailer_anchor_level: int = 0,
    reference_warehouse_levels: list[int] | None = None,
    reference_retailer_levels: list[int] | None = None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    costs = []
    for seed in seeds:
        if is_exact_reference(reference):
            cost = invman_rust.multi_echelon_exact_soft_tree_rollout(
                seed=int(seed),
                **soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=params,
                    include_period_feature=include_period_feature,
                    allocation_mode=allocation_mode,
                    warehouse_base_stock_mode=warehouse_base_stock_mode,
                    policy_feature_mode=policy_feature_mode,
                    policy_action_mode=policy_action_mode,
                    warehouse_anchor_level=warehouse_anchor_level,
                    retailer_anchor_level=retailer_anchor_level,
                    reference_warehouse_levels=reference_warehouse_levels,
                    reference_retailer_levels=reference_retailer_levels,
                ),
            )
        else:
            cost = invman_rust.multi_echelon_soft_tree_rollout(
                seed=int(seed),
                **soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=params,
                    include_period_feature=include_period_feature,
                    allocation_mode=allocation_mode,
                    warehouse_base_stock_mode=warehouse_base_stock_mode,
                    policy_feature_mode=policy_feature_mode,
                    policy_action_mode=policy_action_mode,
                    warehouse_anchor_level=warehouse_anchor_level,
                    retailer_anchor_level=retailer_anchor_level,
                    reference_warehouse_levels=reference_warehouse_levels,
                    reference_retailer_levels=reference_retailer_levels,
                ),
            )
        costs.append(float(cost))
    costs = np.asarray(costs, dtype=np.float64)
    return {
        "mean_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_samples": int(costs.size),
        "costs": costs.tolist(),
    }


def exact_evaluate_soft_tree(
    reference: dict,
    model: SoftTreePolicy,
    *,
    flat_params=None,
    include_period_feature: bool = True,
    allocation_mode: str | None = None,
    warehouse_base_stock_mode: str | None = None,
) -> dict:
    if not is_exact_reference(reference):
        raise ValueError("exact soft-tree evaluation only applies to the exact verifier")
    params = model.get_model_flat_params() if flat_params is None else flat_params
    if allocation_mode is None:
        allocation_mode = benchmark_allocation_mode(reference)
    if warehouse_base_stock_mode is None:
        warehouse_base_stock_mode = benchmark_warehouse_base_stock_mode(reference)
    return dict(
        invman_rust.multi_echelon_exact_evaluate_soft_tree(
            flat_params=np.asarray(params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            min_values=[int(value) for value in model.action_spec["min_values"]],
            max_values=[int(value) for value in model.action_spec["max_values"]],
            action_mode=str(model.action_spec["action_mode"]),
            include_period_feature=bool(include_period_feature),
            warehouse_base_stock_mode=str(warehouse_base_stock_mode),
            allocation_mode=str(allocation_mode),
            temperature=float(model.temperature),
            split_type=str(model.split_type),
            leaf_type=str(model.leaf_type),
            allowed_values=model.action_spec.get("allowed_values"),
        )
    )


def evaluate_soft_tree_policy_exact(
    reference: dict,
    model: SoftTreePolicy,
    *,
    flat_params=None,
    include_period_feature: bool = True,
    allocation_mode: str | None = None,
    warehouse_base_stock_mode: str | None = None,
) -> dict:
    return exact_evaluate_soft_tree(
        reference,
        model,
        flat_params=flat_params,
        include_period_feature=include_period_feature,
        allocation_mode=allocation_mode,
        warehouse_base_stock_mode=warehouse_base_stock_mode,
    )


def evaluate_stationary_policy(
    reference: dict,
    *,
    warehouse_level: int,
    retailer_level: int,
    allocation_mode: str,
    policy_kind: str = "regular_base_stock",
    replications: int | None = None,
    seed: int = 123,
) -> dict:
    if replications is None:
        replications = benchmark_replications(reference)
    result = dict(
        invman_rust.multi_echelon_search_stationary_policy(
            policy_kind=str(policy_kind),
            allocation_mode=str(allocation_mode),
            warehouse_levels=[int(warehouse_level)],
            retailer_levels=[int(retailer_level)],
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
            demand_distribution=str(reference.get("demand_distribution", "normal_rounded_clipped")),
            demand_mean=float(reference.get("demand_mean", 0.0)),
            demand_std=float(reference.get("demand_std", 0.0)),
            horizon=int(benchmark_periods(reference)),
            replications=int(replications),
            seed=int(seed),
            warm_up_periods_ratio=float(benchmark_warm_up_periods_ratio(reference)),
            discount_factor=float(reference.get("discount_factor", 1.0)),
            objective=str(benchmark_rollout_objective(reference)),
            top_k=1,
        )
    )
    best = dict(result["best_result"])
    best["allocation_mode"] = str(allocation_mode)
    best["policy_kind"] = str(policy_kind)
    return best


def search_best_constant_base_stock(
    reference: dict,
    *,
    allocation_mode: str,
    replications: int | None = None,
    seed: int = 123,
    top_k: int = 10,
) -> dict:
    if replications is None:
        replications = benchmark_replications(reference)
    search_horizon = benchmark_periods(reference)
    if not is_exact_reference(reference) and replications != benchmark_replications(reference):
        search_horizon = int(reference.get("benchmark_search_horizon", search_horizon))
    return dict(
        invman_rust.multi_echelon_search_stationary_policy(
            policy_kind="regular_base_stock",
            allocation_mode=str(allocation_mode),
            warehouse_levels=action_grids(reference)[0],
            retailer_levels=action_grids(reference)[1],
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
            demand_distribution=str(reference.get("demand_distribution", "normal_rounded_clipped")),
            demand_mean=float(reference.get("demand_mean", 0.0)),
            demand_std=float(reference.get("demand_std", 0.0)),
            horizon=int(search_horizon),
            replications=int(replications),
            seed=int(seed),
            warm_up_periods_ratio=float(benchmark_warm_up_periods_ratio(reference)),
            discount_factor=float(reference.get("discount_factor", 1.0)),
            objective=str(benchmark_rollout_objective(reference)),
            top_k=int(top_k),
        )
    )


def search_constant_base_stock(
    reference: dict,
    *,
    allocation_mode: str,
    replications: int,
    horizon: int,
    seed: int,
    warehouse_levels: list[int] | None = None,
    retailer_levels: list[int] | None = None,
    top_k: int = 10,
) -> dict:
    if warehouse_levels is None or retailer_levels is None:
        warehouse_levels, retailer_levels = action_grids(reference)
    return dict(
        invman_rust.multi_echelon_search_stationary_policy(
            policy_kind="regular_base_stock",
            allocation_mode=str(allocation_mode),
            warehouse_levels=[int(value) for value in warehouse_levels],
            retailer_levels=[int(value) for value in retailer_levels],
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
            demand_distribution=str(reference.get("demand_distribution", "normal_rounded_clipped")),
            demand_mean=float(reference.get("demand_mean", 0.0)),
            demand_std=float(reference.get("demand_std", 0.0)),
            horizon=int(horizon),
            replications=int(replications),
            seed=int(seed),
            warm_up_periods_ratio=float(benchmark_warm_up_periods_ratio(reference)),
            discount_factor=float(reference.get("discount_factor", 1.0)),
            objective=str(benchmark_rollout_objective(reference)),
            top_k=int(top_k),
        )
    )


def evaluate_van_roy_case_study(
    reference: dict | None = None,
    *,
    allocation_mode: str | None = None,
    replications: int = 1,
    horizon: int | None = None,
    seed: int = 123,
) -> dict:
    if reference is None:
        reference = get_van_roy_case_study()
    if allocation_mode is None:
        allocation_mode = benchmark_allocation_mode(reference)
    levels = [int(value) for value in reference["published_constant_base_stock_levels"]]
    if horizon is None:
        horizon = benchmark_periods(reference)
    result = dict(
        invman_rust.multi_echelon_search_stationary_policy(
            policy_kind="regular_base_stock",
            allocation_mode=str(allocation_mode),
            warehouse_levels=[levels[0]],
            retailer_levels=[levels[1]],
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
            warm_up_periods_ratio=float(reference.get("warm_up_periods_ratio", 0.0)),
            discount_factor=1.0,
            objective=str(reference["rollout_objective"]),
            top_k=1,
        )
    )
    best = dict(result["best_result"])
    best["allocation_mode"] = str(allocation_mode)
    best["published_constant_base_stock_mean_cost"] = float(
        reference["published_constant_base_stock_mean_cost"]
    )
    best["published_constant_base_stock_levels"] = levels
    best["gap_vs_published_cost"] = float(
        best["mean_cost"] - reference["published_constant_base_stock_mean_cost"]
    )
    return best


def implied_target_cost_from_savings_pct(base_cost: float, savings_pct: float | None) -> float | None:
    if savings_pct is None:
        return None
    return float(base_cost * (1.0 - float(savings_pct) / 100.0))


def confidence_half_width(values: Iterable[float], confidence_level: float = 0.95) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    z = float(NormalDist().inv_cdf(0.5 + 0.5 * confidence_level))
    return float(z * np.std(arr, ddof=1) / math.sqrt(arr.size))


def savings_pct(base_cost: float, learned_cost: float) -> float:
    return 100.0 * (float(base_cost) - float(learned_cost)) / float(base_cost)


def savings_pct_samples(base_costs: Iterable[float], learned_costs: Iterable[float]) -> list[float]:
    return [
        savings_pct(base_cost, learned_cost)
        for base_cost, learned_cost in zip(base_costs, learned_costs)
    ]


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
