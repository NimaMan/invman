from __future__ import annotations

from typing import Iterable


def normalize_tree_action_adapter(action_adapter: str) -> str:
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
        raise ValueError(f"Unknown tree action adapter '{action_adapter}'. Expected one of: {valid}")
    return normalized


def _dual_sourcing_target_upper_bound(
    regular_lead_time: int,
    demand_low: int,
    demand_high: int,
    expedited_max_order_size: int,
) -> int:
    mean_demand = 0.5 * (int(demand_low) + int(demand_high))
    upper = int(round((int(regular_lead_time) + 2) * mean_demand + 2 * int(expedited_max_order_size)))
    return max(int(expedited_max_order_size), min(24, upper))


def build_dual_sourcing_control_spec(
    action_adapter: str,
    *,
    regular_lead_time: int,
    demand_low: int,
    demand_high: int,
    regular_max_order_size: int,
    expedited_max_order_size: int,
):
    normalized = normalize_tree_action_adapter(action_adapter)
    target_upper = _dual_sourcing_target_upper_bound(
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


def build_dual_sourcing_action_adapter_config(
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


def _dual_sourcing_action_from_controls(action_adapter: str, controls: list[int], raw_state: list[int], action_adapter_config: dict):
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


def apply_structured_action_adapter(action_adapter: str, controls, normalized_state, action_spec: dict, action_adapter_config: dict | None):
    normalized = normalize_tree_action_adapter(action_adapter)
    if normalized == "identity":
        if action_spec["action_dim"] == 1:
            return int(controls[0])
        return tuple(int(value) for value in controls)

    if action_adapter_config is None:
        raise ValueError(f"action_adapter_config is required for tree action adapter '{normalized}'")

    raw_state = _recover_raw_state(normalized_state, float(action_adapter_config["state_scale"]))
    if normalized.startswith("dual_sourcing_"):
        return _dual_sourcing_action_from_controls(normalized, list(controls), raw_state, action_adapter_config)

    raise NotImplementedError(f"Unsupported tree action adapter: {normalized}")
