from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policy import Policy

import invman_rust


def get_benchmark_reference() -> dict:
    return dict(invman_rust.one_warehouse_multi_retailer_benchmark_reference())


def get_reference(name: str) -> dict:
    return dict(invman_rust.one_warehouse_multi_retailer_get_reference_instance(str(name)))


def list_references() -> list[dict]:
    return [
        dict(reference)
        for reference in invman_rust.one_warehouse_multi_retailer_list_reference_instances()
    ]


def get_primary_reference() -> dict:
    return dict(invman_rust.one_warehouse_multi_retailer_primary_reference_instance())


def get_exact_verification_reference() -> dict:
    return dict(invman_rust.one_warehouse_multi_retailer_exact_verification_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.one_warehouse_multi_retailer_exact_dp_summary())


def exact_evaluate_soft_tree(
    *,
    flat_params,
    input_dim: int,
    depth: int,
    min_values: list[int],
    max_values: list[int],
    action_mode: str,
    allocation_policy: str,
    policy_action_mode: str,
    temperature: float,
    split_type: str,
    leaf_type: str,
    allowed_values=None,
    policy_state_mode: str = "normalized",
) -> dict:
    return dict(
        invman_rust.one_warehouse_multi_retailer_exact_evaluate_soft_tree(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(input_dim),
            depth=int(depth),
            min_values=[int(value) for value in min_values],
            max_values=[int(value) for value in max_values],
            action_mode=str(action_mode),
            allocation_policy=str(allocation_policy),
            policy_action_mode=str(policy_action_mode),
            temperature=float(temperature),
            split_type=str(split_type),
            leaf_type=str(leaf_type),
            allowed_values=allowed_values,
            policy_state_mode=str(policy_state_mode),
        )
    )


def exact_evaluate_echelon_base_stock(
    warehouse_base_stock_level: int,
    retailer_base_stock_levels: list[int],
    *,
    allocation_policy: str,
) -> dict:
    return dict(
        invman_rust.one_warehouse_multi_retailer_exact_evaluate_echelon_base_stock(
            int(warehouse_base_stock_level),
            [int(value) for value in retailer_base_stock_levels],
            str(allocation_policy),
        )
    )


def is_exact_reference(reference: dict) -> bool:
    return "max_action_levels" in reference


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0.0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _rounded_clipped_normal_moments(mean: float, std: float) -> tuple[float, float]:
    if std <= 0.0:
        clipped = max(int(round(mean)), 0)
        return float(clipped), 0.0

    probabilities: list[float] = []
    support: list[int] = []

    # All draws below 0.5 round or clip to zero.
    p_zero = _normal_cdf(0.5, mean, std)
    probabilities.append(max(0.0, min(1.0, p_zero)))
    support.append(0)

    k = 1
    cumulative = probabilities[0]
    while cumulative < 1.0 - 1e-12 and k < 10_000:
        upper = k + 0.5
        lower = k - 0.5
        prob = _normal_cdf(upper, mean, std) - _normal_cdf(lower, mean, std)
        prob = max(0.0, prob)
        if prob > 1e-15:
            probabilities.append(prob)
            support.append(k)
            cumulative += prob
        k += 1

    if cumulative < 1.0:
        probabilities[-1] += 1.0 - cumulative

    support_arr = np.asarray(support, dtype=np.float64)
    prob_arr = np.asarray(probabilities, dtype=np.float64)
    mean_value = float(np.sum(support_arr * prob_arr))
    variance = float(np.sum(((support_arr - mean_value) ** 2) * prob_arr))
    return mean_value, math.sqrt(max(variance, 0.0))


def one_period_demand_moments(reference: dict) -> tuple[list[float], list[float]]:
    if is_exact_reference(reference):
        means: list[float] = []
        stds: list[float] = []
        for support, probabilities in zip(
            reference["demand_supports"],
            reference["demand_probabilities"],
        ):
            support_arr = np.asarray(support, dtype=np.float64)
            prob_arr = np.asarray(probabilities, dtype=np.float64)
            mean = float(np.sum(support_arr * prob_arr))
            variance = float(np.sum(((support_arr - mean) ** 2) * prob_arr))
            means.append(mean)
            stds.append(math.sqrt(max(variance, 0.0)))
        return means, stds

    means: list[float] = []
    stds: list[float] = []
    for kind, param1, param2 in zip(
        reference["demand_kinds"],
        reference["demand_param1"],
        reference["demand_param2"],
    ):
        if kind == "poisson":
            means.append(float(param1))
            stds.append(math.sqrt(float(param1)))
        elif kind == "discrete_uniform":
            low = int(round(param1))
            high = int(round(param2))
            n = high - low + 1
            means.append(0.5 * (low + high))
            stds.append(math.sqrt((n**2 - 1) / 12.0))
        elif kind == "rounded_normal":
            clipped_mean, clipped_std = _rounded_clipped_normal_moments(
                float(param1),
                float(param2),
            )
            means.append(clipped_mean)
            stds.append(clipped_std)
        elif kind == "deterministic":
            means.append(float(param1))
            stds.append(0.0)
        else:
            raise ValueError(f"unsupported demand kind '{kind}'")
    return means, stds


def benchmark_initial_state(reference: dict) -> dict:
    if is_exact_reference(reference):
        return {
            "initial_warehouse_inventory": int(reference["initial_warehouse_inventory"]),
            "initial_warehouse_pipeline": [
                int(value) for value in reference["initial_warehouse_pipeline"]
            ],
            "initial_retailer_inventory": [
                int(value) for value in reference["initial_retailer_inventory"]
            ],
            "initial_retailer_pipeline": [
                [int(value) for value in row]
                for row in reference["initial_retailer_pipeline"]
            ],
            "initial_state_rule": "repo_exact_reference",
        }

    means, _ = one_period_demand_moments(reference)
    warehouse_mean = int(round(sum(means)))
    retailer_inventory = [int(round(mean)) for mean in means]
    return {
        "initial_warehouse_inventory": warehouse_mean,
        "initial_warehouse_pipeline": [
            warehouse_mean for _ in range(int(reference["warehouse_lead_time"]))
        ],
        "initial_retailer_inventory": retailer_inventory,
        "initial_retailer_pipeline": [
            [retailer_inventory[idx] for _ in range(int(lead_time))]
            for idx, lead_time in enumerate(reference["retailer_lead_times"])
        ],
        "initial_state_rule": "mean_filled_pipeline_warm_start",
    }


def benchmark_discount_factor(reference: dict) -> float:
    return float(reference["discount_factor"]) if is_exact_reference(reference) else 1.0


def benchmark_periods(reference: dict) -> int:
    return int(reference["periods"]) if is_exact_reference(reference) else int(
        reference["benchmark_periods"]
    )


def benchmark_replications(reference: dict) -> int:
    return 1 if is_exact_reference(reference) else int(reference["benchmark_replications"])


def policy_state_input_dim(reference: dict, policy_state_mode: str = "normalized") -> int:
    normalized_dim = (
        1
        + int(reference["warehouse_lead_time"])
        + len(reference["retailer_lead_times"])
        + sum(int(value) for value in reference["retailer_lead_times"])
        + 2
    )
    if policy_state_mode in ("normalized", "default"):
        return normalized_dim
    if policy_state_mode in ("absolute_augmented", "augmented", "absolute"):
        return normalized_dim + 2 + len(reference["retailer_lead_times"])
    raise ValueError(
        f"unsupported policy_state_mode '{policy_state_mode}'; expected normalized or absolute_augmented"
    )


def is_symmetric_retailer_case(reference: dict) -> bool:
    if is_exact_reference(reference):
        return (
            len(set(reference["retailer_lead_times"])) == 1
            and len(set(reference["holding_cost_retailers"])) == 1
            and len(set(reference["penalty_costs_retailers"])) == 1
            and all(
                reference["demand_supports"][idx] == reference["demand_supports"][0]
                and reference["demand_probabilities"][idx]
                == reference["demand_probabilities"][0]
                for idx in range(1, len(reference["retailer_lead_times"])))
        )

    return (
        len(set(reference["retailer_lead_times"])) == 1
        and len(set(reference["holding_cost_retailers"])) == 1
        and len(set(reference["penalty_costs_retailers"])) == 1
        and all(
            reference["demand_kinds"][idx] == reference["demand_kinds"][0]
            and float(reference["demand_param1"][idx]) == float(reference["demand_param1"][0])
            and float(reference["demand_param2"][idx]) == float(reference["demand_param2"][0])
            for idx in range(1, len(reference["retailer_lead_times"])))
        )


def echelon_base_stock_search_bounds(reference: dict) -> dict:
    means, stds = one_period_demand_moments(reference)
    retailer_bounds = []
    for mean, std, lead_time in zip(means, stds, reference["retailer_lead_times"]):
        lead_periods = int(lead_time) + 1
        lower = int(math.floor(mean * lead_periods))
        upper = int(math.ceil(mean * lead_periods + 3.0 * std * math.sqrt(lead_periods)))
        retailer_bounds.append((max(0, lower), max(0, upper)))

    system_mean = float(sum(means))
    system_variance = float(sum(std**2 for std in stds))
    cumulative_lead_periods = int(reference["warehouse_lead_time"]) + max(
        int(value) for value in reference["retailer_lead_times"]
    ) + 1
    warehouse_lower = int(math.floor(system_mean * cumulative_lead_periods))
    warehouse_upper = int(
        math.ceil(
            system_mean * cumulative_lead_periods
            + 3.0 * math.sqrt(system_variance * cumulative_lead_periods)
        )
    )
    return {
        "warehouse": (max(0, warehouse_lower), max(0, warehouse_upper)),
        "retailers": retailer_bounds,
        "symmetric_retailers": bool(is_symmetric_retailer_case(reference)),
    }


def uses_kaynov_k_search(reference: dict) -> bool:
    return (not is_exact_reference(reference)) and str(reference.get("name")) == "kaynov2024_instance_14"


def _retailer_targets_from_k(reference: dict, k_value: float) -> list[int]:
    means, stds = one_period_demand_moments(reference)
    targets: list[int] = []
    for mean, std, lead_time in zip(means, stds, reference["retailer_lead_times"]):
        lead_periods = int(lead_time) + 1
        level = mean * lead_periods + k_value * std * math.sqrt(lead_periods)
        targets.append(max(0, int(round(level))))
    return targets


def kaynov_instance14_k_candidates(reference: dict) -> list[float]:
    if not uses_kaynov_k_search(reference):
        raise ValueError("k-candidate generation is only defined for Kaynov instance 14")

    means, stds = one_period_demand_moments(reference)
    breakpoints = {0.0, 3.0}
    for mean, std, lead_time in zip(means, stds, reference["retailer_lead_times"]):
        lead_periods = int(lead_time) + 1
        intercept = mean * lead_periods
        slope = std * math.sqrt(lead_periods)
        if slope <= 1e-12:
            continue
        lower_level = int(math.floor(intercept))
        upper_level = int(math.ceil(intercept + 3.0 * slope))
        for level in range(lower_level - 1, upper_level + 2):
            cutoff = (level + 0.5 - intercept) / slope
            if 0.0 <= cutoff <= 3.0:
                breakpoints.add(float(cutoff))

    ordered = sorted(breakpoints)
    candidates = {0.0, 3.0}
    for left, right in zip(ordered[:-1], ordered[1:]):
        midpoint = 0.5 * (left + right)
        if 0.0 <= midpoint <= 3.0:
            candidates.add(midpoint)

    candidates_with_vectors: dict[tuple[int, ...], float] = {}
    for candidate in sorted(candidates):
        vector = tuple(_retailer_targets_from_k(reference, candidate))
        candidates_with_vectors.setdefault(vector, candidate)
    return sorted(candidates_with_vectors.values())


def evaluate_echelon_base_stock_policy(
    reference: dict,
    *,
    warehouse_base_stock_level: int,
    retailer_base_stock_levels: list[int],
    allocation_policy: str,
    replications: int | None = None,
    seed: int = 123,
) -> dict:
    if is_exact_reference(reference) and replications is None:
        result = exact_evaluate_echelon_base_stock(
            warehouse_base_stock_level,
            retailer_base_stock_levels,
            allocation_policy=allocation_policy,
        )
        return {
            "warehouse_base_stock_level": int(warehouse_base_stock_level),
            "retailer_base_stock_levels": [
                int(value) for value in retailer_base_stock_levels
            ],
            "allocation_policy": str(allocation_policy),
            "mean_cost": float(result["discounted_cost"]),
            "first_action": list(result["first_action"]),
            "evaluation_mode": "exact",
        }

    initial_state = benchmark_initial_state(reference)
    if is_exact_reference(reference):
        demand_kinds, demand_param1, demand_param2 = _exact_support_to_rollout_models(reference)
    else:
        demand_kinds = [str(value) for value in reference["demand_kinds"]]
        demand_param1 = [float(value) for value in reference["demand_param1"]]
        demand_param2 = [float(value) for value in reference["demand_param2"]]

    mean_cost, cost_std = invman_rust.one_warehouse_multi_retailer_simulate_policy(
        policy_name="echelon_base_stock",
        params=[float(warehouse_base_stock_level)]
        + [float(value) for value in retailer_base_stock_levels],
        initial_warehouse_inventory=int(initial_state["initial_warehouse_inventory"]),
        initial_warehouse_pipeline=initial_state["initial_warehouse_pipeline"],
        initial_retailer_inventory=initial_state["initial_retailer_inventory"],
        initial_retailer_pipeline=initial_state["initial_retailer_pipeline"],
        periods=benchmark_periods(reference),
        replications=int(
            benchmark_replications(reference)
            if replications is None
            else replications
        ),
        seed=int(seed),
        demand_kinds=demand_kinds,
        demand_param1=demand_param1,
        demand_param2=demand_param2,
        holding_cost_warehouse=float(reference["holding_cost_warehouse"]),
        holding_cost_retailers=[
            float(value) for value in reference["holding_cost_retailers"]
        ],
        penalty_costs_retailers=[
            float(value) for value in reference["penalty_costs_retailers"]
        ],
        customer_behavior=str(reference["customer_behavior"]),
        emergency_shipment_probability=float(reference["emergency_shipment_probability"]),
        discount_factor=benchmark_discount_factor(reference),
        allocation_policy=str(allocation_policy),
    )
    return {
        "warehouse_base_stock_level": int(warehouse_base_stock_level),
        "retailer_base_stock_levels": [int(value) for value in retailer_base_stock_levels],
        "allocation_policy": str(allocation_policy),
        "mean_cost": float(mean_cost),
        "cost_std": float(cost_std),
        "num_samples": int(
            benchmark_replications(reference) if replications is None else replications
        ),
        "initial_state": initial_state,
        "evaluation_mode": "simulation",
    }


def search_best_echelon_base_stock(
    reference: dict,
    *,
    allocation_policy: str,
    replications: int | None = None,
    seed: int = 123,
) -> dict:
    bounds = echelon_base_stock_search_bounds(reference)
    warehouse_levels = range(bounds["warehouse"][0], bounds["warehouse"][1] + 1)
    if uses_kaynov_k_search(reference):
        candidates = (
            (warehouse_level, _retailer_targets_from_k(reference, k_value), k_value)
            for warehouse_level in warehouse_levels
            for k_value in kaynov_instance14_k_candidates(reference)
        )
    elif bounds["symmetric_retailers"]:
        common_bounds = bounds["retailers"][0]
        candidates = (
            (
                warehouse_level,
                [retailer_level] * len(reference["retailer_lead_times"]),
                None,
            )
            for warehouse_level in warehouse_levels
            for retailer_level in range(common_bounds[0], common_bounds[1] + 1)
        )
    else:
        retailer_grids = [
            range(lower, upper + 1) for lower, upper in bounds["retailers"]
        ]
        candidates = (
            (warehouse_level, list(retailer_levels), None)
            for warehouse_level in warehouse_levels
            for retailer_levels in product(*retailer_grids)
        )

    best = None
    for warehouse_level, retailer_levels, k_value in candidates:
        evaluation = evaluate_echelon_base_stock_policy(
            reference,
            warehouse_base_stock_level=warehouse_level,
            retailer_base_stock_levels=retailer_levels,
            allocation_policy=allocation_policy,
            replications=replications,
            seed=seed,
        )
        if best is None or float(evaluation["mean_cost"]) < float(best["mean_cost"]):
            best = evaluation

    if best is None:
        raise RuntimeError("heuristic search produced no candidates")

    best["search_bounds"] = bounds
    best["search_seed"] = int(seed)
    best["search_protocol"] = (
        "kaynov_instance14_z0_k"
        if uses_kaynov_k_search(reference)
        else "full_cartesian_symmetric_reduction"
    )
    if uses_kaynov_k_search(reference):
        matching_k_values = [
            k_value
            for k_value in kaynov_instance14_k_candidates(reference)
            if _retailer_targets_from_k(reference, k_value) == best["retailer_base_stock_levels"]
        ]
        best["k_value_candidates"] = [float(value) for value in matching_k_values]
    if replications is not None:
        best["search_replications"] = int(replications)
    return best


def policy_action_mode_for_reference(reference: dict) -> str:
    return (
        "symmetric_echelon_targets"
        if is_symmetric_retailer_case(reference)
        else "echelon_targets"
    )


def _exact_support_to_rollout_models(reference: dict) -> tuple[list[str], list[float], list[float]]:
    demand_kinds: list[str] = []
    demand_param1: list[float] = []
    demand_param2: list[float] = []
    for support, probabilities in zip(
        reference["demand_supports"],
        reference["demand_probabilities"],
    ):
        if support == [0, 1] and probabilities == [0.5, 0.5]:
            demand_kinds.append("discrete_uniform")
            demand_param1.append(0.0)
            demand_param2.append(1.0)
            continue
        raise ValueError(
            "the exact verification reference uses an unsupported demand support for rollout-backed tree training"
        )
    return demand_kinds, demand_param1, demand_param2


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    policy_action_mode: str = "direct_orders",
    policy_state_mode: str = "normalized",
) -> Policy:
    input_dim = policy_state_input_dim(reference, policy_state_mode)
    if policy_action_mode == "symmetric_echelon_targets":
        if not is_symmetric_retailer_case(reference):
            raise ValueError("symmetric_echelon_targets requires a symmetric retailer reference")
        if is_exact_reference(reference):
            warehouse_levels = list(range(int(reference["max_action_levels"][0]) + 1))
            retailer_levels = list(range(int(reference["max_action_levels"][1]) + 1))
        else:
            bounds = echelon_base_stock_search_bounds(reference)
            warehouse_levels = list(
                range(int(bounds["warehouse"][0]), int(bounds["warehouse"][1]) + 1)
            )
            retailer_lower, retailer_upper = bounds["retailers"][0]
            retailer_levels = list(range(int(retailer_lower), int(retailer_upper) + 1))
        return Policy(
            backbone="soft_tree",
            input_dim=input_dim,
            control_dim=2,
            control_mode="discrete_grid",
            min_values=[int(warehouse_levels[0]), int(retailer_levels[0])],
            max_values=[int(warehouse_levels[-1]), int(retailer_levels[-1])],
            allowed_values=[warehouse_levels, retailer_levels],
            depth=int(depth),
            temperature=float(temperature),
            split_type=str(split_type),
            leaf_type=str(leaf_type),
            state_normalizer="identity",
            state_scale=None,
            state_feature_mode=str(policy_state_mode),
        )

    if is_exact_reference(reference):
        max_values = [int(value) for value in reference["max_action_levels"]]
    else:
        bounds = echelon_base_stock_search_bounds(reference)
        max_values = [int(bounds["warehouse"][1])] + [
            int(upper) for _, upper in bounds["retailers"]
        ]
    if policy_action_mode == "echelon_targets_with_alloc_targets":
        retailer_max_values = max_values[1:]
        max_values = [max_values[0]] + retailer_max_values + retailer_max_values
        control_dim = 1 + 2 * len(reference["retailer_lead_times"])
    else:
        control_dim = len(reference["retailer_lead_times"]) + 1
    return Policy(
        backbone="soft_tree",
        input_dim=input_dim,
        control_dim=control_dim,
        control_mode="vector_quantity",
        min_values=[0] * control_dim,
        max_values=max_values,
        allowed_values=None,
        depth=int(depth),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        state_normalizer="identity",
        state_scale=None,
        state_feature_mode=str(policy_state_mode),
    )


def soft_tree_rollout_kwargs(
    reference: dict,
    model: Policy,
    *,
    flat_params,
    allocation_policy: str,
    policy_action_mode: str = "direct_orders",
) -> dict:
    initial_state = benchmark_initial_state(reference)
    if is_exact_reference(reference):
        demand_kinds, demand_param1, demand_param2 = _exact_support_to_rollout_models(reference)
    else:
        demand_kinds = [str(value) for value in reference["demand_kinds"]]
        demand_param1 = [float(value) for value in reference["demand_param1"]]
        demand_param2 = [float(value) for value in reference["demand_param2"]]

    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.min_values],
        "max_values": [int(value) for value in model.max_values],
        "action_mode": str(model.control_mode),
        "initial_warehouse_inventory": int(initial_state["initial_warehouse_inventory"]),
        "initial_warehouse_pipeline": initial_state["initial_warehouse_pipeline"],
        "initial_retailer_inventory": initial_state["initial_retailer_inventory"],
        "initial_retailer_pipeline": initial_state["initial_retailer_pipeline"],
        "demand_kinds": demand_kinds,
        "demand_param1": demand_param1,
        "demand_param2": demand_param2,
        "holding_cost_warehouse": float(reference["holding_cost_warehouse"]),
        "holding_cost_retailers": [
            float(value) for value in reference["holding_cost_retailers"]
        ],
        "penalty_costs_retailers": [
            float(value) for value in reference["penalty_costs_retailers"]
        ],
        "customer_behavior": str(reference["customer_behavior"]),
        "periods": benchmark_periods(reference),
        "emergency_shipment_probability": float(reference["emergency_shipment_probability"]),
        "discount_factor": benchmark_discount_factor(reference),
        "allocation_policy": str(allocation_policy),
        "policy_action_mode": str(policy_action_mode),
        "retailer_target_inventory_positions": None,
        "temperature": float(model.temperature),
        "split_type": str(model.split_type),
        "leaf_type": str(model.leaf_type),
        "allowed_values": model.allowed_values,
        "policy_state_mode": str(getattr(model, "state_feature_mode", "normalized")),
    }


def evaluate_soft_tree_policy(
    reference: dict,
    model: Policy,
    seeds: Iterable[int],
    *,
    allocation_policy: str,
    policy_action_mode: str = "direct_orders",
    flat_params=None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    costs = []
    for seed in seeds:
        discounted_cost = invman_rust.one_warehouse_multi_retailer_soft_tree_rollout(
            seed=int(seed),
            **soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=params,
                allocation_policy=allocation_policy,
                policy_action_mode=policy_action_mode,
            ),
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


def published_cost(published_row: dict | None) -> float | None:
    if published_row is None:
        return None
    return float(-published_row["mean_cost"])


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
