from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policies.soft_tree import SoftTreePolicy

import invman_rust


MEDIUM_REFERENCE_INSTANCE_NAME = "de_moor2022_m4_exp6_l2_cp7_fifo"


def get_reference(name: str) -> dict:
    return dict(invman_rust.perishable_inventory_get_reference_instance(name))


def primary_reference_name() -> str:
    return str(invman_rust.perishable_inventory_primary_reference_instance_name())


def zero_state(reference: dict) -> tuple[list[int], list[int]]:
    return (
        [0 for _ in range(int(reference["shelf_life"]))],
        [0 for _ in range(max(int(reference["lead_time"]) - 1, 0))],
    )

def evaluate_heuristic_policy(
    reference: dict,
    policy_name: str,
    params,
    seeds: Iterable[int],
    *,
    horizon: int | None = None,
) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_policy_discounted_return_summary(
            policy_name=policy_name,
            params=list(params) if isinstance(params, tuple) else [int(params)],
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            horizon=int(reference["horizon"] if horizon is None else horizon),
            seeds=[int(seed) for seed in seeds],
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(reference["demand_mean"]),
            demand_cov=float(reference["demand_cov"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
            issuing_policy=str(reference["issuing_policy"]),
        )
    )


def search_best_base_stock(reference: dict, seeds: Iterable[int], *, horizon: int | None = None) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_base_stock_search_discounted_return_summary(
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            horizon=int(reference["horizon"] if horizon is None else horizon),
            seeds=[int(seed) for seed in seeds],
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(reference["demand_mean"]),
            demand_cov=float(reference["demand_cov"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            position_upper_bound=int(reference["max_order_size"]),
            procurement_cost=float(reference["procurement_cost"]),
            warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
            issuing_policy=str(reference["issuing_policy"]),
            top_k=12,
        )
    )


def search_best_base_stock_from_demands(
    reference: dict,
    demands: Iterable[int],
    *,
    demand_mean: float,
) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    best, top = invman_rust.perishable_inventory_base_stock_search_from_demands(
        on_hand=on_hand,
        pipeline_orders=pipeline_orders,
        demands=[int(value) for value in demands],
        lead_time=int(reference["lead_time"]),
        max_order_size=int(reference["max_order_size"]),
        demand_mean=float(demand_mean),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
        waste_cost=float(reference["waste_cost"]),
        position_upper_bound=int(reference["max_order_size"]),
        procurement_cost=float(reference["procurement_cost"]),
        warm_up_periods_ratio=0.0,
        issuing_policy=str(reference["issuing_policy"]),
        top_k=12,
    )
    return {
        "best": {
            "params": [int(best[0])],
            "mean_period_cost": float(best[1]),
        },
        "top": [
            {
                "params": [int(params)],
                "mean_period_cost": float(mean_period_cost),
            }
            for params, mean_period_cost in top
        ],
    }


def search_best_bsp_low_ew(reference: dict, seeds: Iterable[int], *, horizon: int | None = None) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_bsp_low_ew_search_discounted_return_summary(
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            horizon=int(reference["horizon"] if horizon is None else horizon),
            seeds=[int(seed) for seed in seeds],
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(reference["demand_mean"]),
            demand_cov=float(reference["demand_cov"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            position_upper_bound=int(reference["max_order_size"]),
            procurement_cost=float(reference["procurement_cost"]),
            warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
            issuing_policy=str(reference["issuing_policy"]),
            top_k=12,
        )
    )


def search_best_bsp_low_ew_from_demands(
    reference: dict,
    demands: Iterable[int],
    *,
    demand_mean: float,
) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    best, top = invman_rust.perishable_inventory_bsp_low_ew_search_from_demands(
        on_hand=on_hand,
        pipeline_orders=pipeline_orders,
        demands=[int(value) for value in demands],
        lead_time=int(reference["lead_time"]),
        max_order_size=int(reference["max_order_size"]),
        demand_mean=float(demand_mean),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
        waste_cost=float(reference["waste_cost"]),
        position_upper_bound=int(reference["max_order_size"]),
        procurement_cost=float(reference["procurement_cost"]),
        warm_up_periods_ratio=0.0,
        issuing_policy=str(reference["issuing_policy"]),
        top_k=12,
    )
    return {
        "best": {
            "params": [int(best[0]), int(best[1]), int(best[2])],
            "mean_period_cost": float(best[3]),
        },
        "top": [
            {
                "params": [int(s1), int(s2), int(b)],
                "mean_period_cost": float(mean_period_cost),
            }
            for s1, s2, b, mean_period_cost in top
        ],
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
        input_dim=int(reference["shelf_life"]) + int(reference["lead_time"]) - 1,
        action_spec={
            "action_dim": 1,
            "action_mode": "scalar_quantity",
            "min_values": [0],
            "max_values": [int(reference["max_order_size"])],
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
    horizon: int,
) -> dict:
    return {
        "flat_params": np.asarray(flat_params, dtype=np.float32).tolist(),
        "input_dim": int(model.input_dim),
        "depth": int(model.depth),
        "min_values": [int(value) for value in model.action_spec["min_values"]],
        "max_values": [int(value) for value in model.action_spec["max_values"]],
        "action_mode": str(model.action_spec["action_mode"]),
        "demand_mean": float(reference["demand_mean"]),
        "demand_cov": float(reference["demand_cov"]),
        "shelf_life": int(reference["shelf_life"]),
        "lead_time": int(reference["lead_time"]),
        "holding_cost": float(reference["holding_cost"]),
        "shortage_cost": float(reference["shortage_cost"]),
        "waste_cost": float(reference["waste_cost"]),
        "procurement_cost": float(reference["procurement_cost"]),
        "horizon": int(horizon),
        "warm_up_periods_ratio": float(reference["warm_up_periods_ratio"]),
        "temperature": float(model.temperature),
        "split_type": str(model.split_type),
        "leaf_type": str(model.leaf_type),
        "issuing_policy": str(reference["issuing_policy"]),
        "allowed_values": model.action_spec.get("allowed_values"),
    }


def evaluate_soft_tree_policy(
    reference: dict,
    model: SoftTreePolicy,
    seeds: Iterable[int],
    *,
    flat_params=None,
    horizon: int | None = None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    returns = []
    for seed in seeds:
        discounted_return = invman_rust.perishable_inventory_soft_tree_discounted_return(
            seed=int(seed),
            **soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=params,
                horizon=int(reference["horizon"] if horizon is None else horizon),
            ),
        )
        returns.append(float(discounted_return))
    returns = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "num_seeds": int(returns.size),
    }


def evaluate_heuristic_trace_summary(
    reference: dict,
    policy_name: str,
    params,
    demands: Iterable[int],
    *,
    demand_mean: float,
) -> dict:
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_policy_trace_summary_from_demands(
            policy_name=policy_name,
            params=list(params) if isinstance(params, tuple) else [int(value) for value in params],
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            demands=[int(value) for value in demands],
            lead_time=int(reference["lead_time"]),
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(demand_mean),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            issuing_policy=str(reference["issuing_policy"]),
        )
    )


def evaluate_soft_tree_trace_summary(
    reference: dict,
    model: SoftTreePolicy,
    demands: Iterable[int],
    *,
    demand_mean: float,
    flat_params=None,
) -> dict:
    params = model.get_model_flat_params() if flat_params is None else flat_params
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_soft_tree_trace_summary_from_demands(
            flat_params=np.asarray(params, dtype=np.float32).tolist(),
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            demands=[int(value) for value in demands],
            demand_mean=float(demand_mean),
            **{
                key: value
                for key, value in soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=params,
                    horizon=int(reference["horizon"]),
                ).items()
                if key
                not in {
                    "flat_params",
                    "demand_mean",
                    "demand_cov",
                    "shelf_life",
                    "lead_time",
                    "horizon",
                    "warm_up_periods_ratio",
                }
            },
        )
    )


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
