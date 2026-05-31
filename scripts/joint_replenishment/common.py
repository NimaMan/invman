from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from invman.policy import Policy

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


def newsvendor_item_targets(reference: dict) -> list[int]:
    # Order-up-to S_i per item from the single-period newsvendor critical ratio
    # cr_i = b_i / (b_i + h_i) over the uniform per-period demand U[low, high].
    # This is the target both carried heuristics (MOQ, DYN-OUT) are evaluated at,
    # matching scripts/joint_replenishment/benchmark_vanvuchelen_settings.py.
    targets: list[int] = []
    for high, low, holding, shortage in zip(
        reference["demand_highs"],
        reference["demand_lows"],
        reference["holding_costs"],
        reference["shortage_costs"],
    ):
        cr = float(shortage) / (float(shortage) + float(holding))
        support = int(high) - int(low) + 1
        target = next(
            (
                int(low) + offset
                for offset in range(0, support)
                if (offset + 1) / support >= cr - 1e-12
            ),
            int(high),
        )
        targets.append(int(target))
    return targets


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
        initial_inventory_levels=[
            int(value)
            for value in reference.get(
                "initial_inventory_levels", [0] * len(reference["demand_highs"])
            )
        ],
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
        discount_factor=float(reference.get("discount_factor", 0.99)),
    )
    return {
        "mean_cost": float(mean_cost),
        "cost_std": float(cost_std),
        "num_samples": int(replications),
        "params": [float(value) for value in params],
    }


def _max_order_quantities(reference: dict, *, action_box: str = "wide", cap_slack: int = 1) -> list[int]:
    # The 16 Table-2 reference instances carry only the cost/demand structure; the
    # exact-verification reference additionally carries explicit caps. When absent,
    # derive the per-item soft-tree action box.
    #
    # ACTION-BOX DESIGN (the high-cost-setting recovery lever, see
    # autoresearch/program_joint_replenishment.md). The `vector_quantity` action mode
    # decodes a tree output into an integer order in [0, max_value_i]. With the default
    # "wide" box (2*truck_capacity = 12 per item) the decode resolution is coarse exactly
    # in the region the optimal base-stock policy operates (orders rarely exceed the
    # per-item newsvendor target = demand_high_i for the high-cost h=5,b=95 family). The
    # "basestock" box CAPS each item at its newsvendor target + cap_slack, so the same
    # tree-output range maps onto a tight band around the base-stock order -- a
    # base-stock-anchored action adapter implemented at the Python action-box layer (the
    # Rust decoder is read-only here). This both finens the decode resolution around the
    # optimal order and makes the zero/MOQ warm-start anchor land near sane orders instead
    # of saturating the box.
    if reference.get("max_order_quantities") is not None:
        return [int(value) for value in reference["max_order_quantities"]]
    num_items = int(reference.get("num_items", len(reference["demand_highs"])))
    if str(action_box) == "basestock":
        targets = newsvendor_item_targets(reference)
        return [int(targets[i]) + int(cap_slack) for i in range(num_items)]
    cap = 2 * int(reference["truck_capacity"])
    return [cap for _ in range(num_items)]


def build_soft_tree_model(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    action_box: str = "wide",
    cap_slack: int = 1,
) -> Policy:
    caps = _max_order_quantities(reference, action_box=str(action_box), cap_slack=int(cap_slack))
    num_items = len(caps)
    return Policy(
        backbone="soft_tree",
        input_dim=num_items + 2,
        control_dim=num_items,
        control_mode="vector_quantity",
        min_values=tuple(0 for _ in range(num_items)),
        max_values=tuple(int(value) for value in caps),
        allowed_values=None,
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
        "initial_inventory_levels": [
            int(value)
            for value in reference.get(
                "initial_inventory_levels", [0] * len(reference["demand_highs"])
            )
        ],
        "demand_lows": [int(value) for value in reference["demand_lows"]],
        "demand_highs": [int(value) for value in reference["demand_highs"]],
        "truck_capacity": int(reference["truck_capacity"]),
        "minor_order_costs": [float(value) for value in reference["minor_order_costs"]],
        "major_order_cost": float(reference["major_order_cost"]),
        "holding_costs": [float(value) for value in reference["holding_costs"]],
        "shortage_costs": [float(value) for value in reference["shortage_costs"]],
        "periods": int(reference["periods"]),
        "discount_factor": float(reference.get("discount_factor", 0.99)),
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
