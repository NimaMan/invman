"""Policy support and structured action maps for dual sourcing."""

from __future__ import annotations

from typing import Iterable

from invman.policies.common import normalize_action_spec

SUPPORTED_POLICY_BACKBONES = ("linear", "nn", "soft_tree")


def normalize_action_adapter(action_adapter: str) -> str:
    aliases = {
        "identity": "identity",
        "direct": "identity",
        "direct_orders": "identity",
        "dual_sourcing_single_index_targets": "dual_sourcing_single_index_targets",
        "single_index_targets": "dual_sourcing_single_index_targets",
        "dual_sourcing_dual_index_targets": "dual_sourcing_dual_index_targets",
        "dual_index_targets": "dual_sourcing_dual_index_targets",
        "dual_sourcing_capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
        "capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
        "dual_sourcing_base_surge_targets": "dual_sourcing_base_surge_targets",
        "base_surge_targets": "dual_sourcing_base_surge_targets",
    }
    normalized = aliases.get(action_adapter)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown action adapter '{action_adapter}'. Expected one of: {valid}")
    return normalized


def _target_upper_bound(
    regular_lead_time: int,
    demand_low: int,
    demand_high: int,
    expedited_max_order_size: int,
) -> int:
    mean_demand = 0.5 * (int(demand_low) + int(demand_high))
    upper = int(round((int(regular_lead_time) + 2) * mean_demand + 2 * int(expedited_max_order_size)))
    return max(int(expedited_max_order_size), min(24, upper))


def build_control_spec(
    action_adapter: str,
    *,
    regular_lead_time: int,
    demand_low: int,
    demand_high: int,
    regular_max_order_size: int,
    expedited_max_order_size: int,
):
    normalized = normalize_action_adapter(action_adapter)
    target_upper = _target_upper_bound(
        regular_lead_time=regular_lead_time,
        demand_low=demand_low,
        demand_high=demand_high,
        expedited_max_order_size=expedited_max_order_size,
    )
    if normalized == "identity":
        return {
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [int(regular_max_order_size), int(expedited_max_order_size)],
            "allowed_values": None,
        }
    if normalized in {
        "dual_sourcing_single_index_targets",
        "dual_sourcing_dual_index_targets",
    }:
        return {
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [int(target_upper), int(target_upper)],
            "allowed_values": None,
        }
    if normalized == "dual_sourcing_capped_dual_index_targets":
        return {
            "action_dim": 3,
            "action_mode": "vector_quantity",
            "min_values": [0, 0, 0],
            "max_values": [int(target_upper), int(target_upper), int(regular_max_order_size)],
            "allowed_values": None,
        }
    if normalized == "dual_sourcing_base_surge_targets":
        return {
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [int(target_upper), int(regular_max_order_size)],
            "allowed_values": None,
        }
    raise NotImplementedError(f"Unsupported dual-sourcing action adapter: {normalized}")


def build_action_adapter_config(
    *,
    regular_max_order_size: int,
    expedited_max_order_size: int,
    state_scale: float,
):
    return {
        "regular_max_order_size": int(regular_max_order_size),
        "expedited_max_order_size": int(expedited_max_order_size),
        "state_scale": float(state_scale),
    }


def _recover_raw_state(normalized_state: Iterable[float], state_scale: float):
    return [int(round(float(value) * float(state_scale))) for value in normalized_state]


def _action_from_controls(action_adapter: str, controls: list[int], raw_state: list[int], action_adapter_config: dict):
    expedited_inventory_position = int(raw_state[0])
    regular_inventory_position = int(sum(raw_state))
    max_regular = int(action_adapter_config["regular_max_order_size"])
    max_expedited = int(action_adapter_config["expedited_max_order_size"])

    if action_adapter == "dual_sourcing_single_index_targets":
        s_e = int(controls[0])
        s_r = max(int(controls[1]), s_e)
        expedited = min(max(0, s_e - regular_inventory_position), max_expedited)
        regular = min(max(0, s_r - regular_inventory_position - expedited), max_regular)
        return regular, expedited

    if action_adapter == "dual_sourcing_dual_index_targets":
        s_e = int(controls[0])
        s_r = max(int(controls[1]), s_e)
        expedited = min(max(0, s_e - expedited_inventory_position), max_expedited)
        regular = min(max(0, s_r - regular_inventory_position - expedited), max_regular)
        return regular, expedited

    if action_adapter == "dual_sourcing_capped_dual_index_targets":
        s_e = int(controls[0])
        s_r = max(int(controls[1]), s_e)
        cap_r = int(controls[2])
        expedited = min(max(0, s_e - expedited_inventory_position), max_expedited)
        desired_regular = max(0, s_r - regular_inventory_position - expedited)
        regular = min(desired_regular, cap_r, max_regular)
        return regular, expedited

    if action_adapter == "dual_sourcing_base_surge_targets":
        surge_level = int(controls[0])
        regular_qty = int(controls[1])
        expedited = min(max(0, surge_level - expedited_inventory_position), max_expedited)
        regular = min(max(0, regular_qty), max_regular)
        return regular, expedited

    raise NotImplementedError(f"Unsupported dual-sourcing action adapter: {action_adapter}")


def apply_action_adapter(action_adapter: str, controls, normalized_state, action_spec: dict, action_adapter_config: dict | None):
    normalized = normalize_action_adapter(action_adapter)
    if normalized == "identity":
        if action_spec["action_dim"] == 1:
            return int(controls[0])
        return tuple(int(value) for value in controls)

    if action_adapter_config is None:
        raise ValueError(f"action_adapter_config is required for action adapter '{normalized}'")

    raw_state = _recover_raw_state(normalized_state, float(action_adapter_config["state_scale"]))
    return _action_from_controls(normalized, list(controls), raw_state, action_adapter_config)


def build_policy_context(args, env):
    from invman.policies.registry import get_policy_spec

    action_spec = normalize_action_spec(getattr(env, "action_spec", None))
    action_adapter = get_policy_spec(args).action_adapter
    control_spec = None
    action_adapter_config = None
    if action_adapter != "identity":
        control_spec = build_control_spec(
            action_adapter,
            regular_lead_time=int(env.regular_lead_time),
            demand_low=int(env.demand_low),
            demand_high=int(env.demand_high),
            regular_max_order_size=int(env.regular_max_order_size),
            expedited_max_order_size=int(env.expedited_max_order_size),
        )
        action_adapter_config = build_action_adapter_config(
            regular_max_order_size=int(env.regular_max_order_size),
            expedited_max_order_size=int(env.expedited_max_order_size),
            state_scale=float(max(1, env.regular_max_order_size + env.expedited_max_order_size)),
        )
    return {
        "supported_policy_backbones": SUPPORTED_POLICY_BACKBONES,
        "action_spec": action_spec,
        "control_spec": control_spec,
        "action_adapter": action_adapter,
        "action_adapter_config": action_adapter_config,
    }
