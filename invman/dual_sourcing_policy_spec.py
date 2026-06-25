"""Structured control bounds for dual-sourcing learned policies."""

from __future__ import annotations


def normalize_action_adapter(action_adapter: str) -> str:
    aliases = {
        "identity": "identity",
        "direct": "identity",
        "direct_orders": "identity",
        "dual_sourcing_single_index_targets": "dual_sourcing_single_index_targets",
        "single_index_targets": "dual_sourcing_single_index_targets",
        "dual_sourcing_dual_index_targets": "dual_sourcing_dual_index_targets",
        "dual_index_targets": "dual_sourcing_dual_index_targets",
        "dual_sourcing_dual_index_delta_targets": "dual_sourcing_dual_index_delta_targets",
        "dual_index_delta_targets": "dual_sourcing_dual_index_delta_targets",
        "dual_sourcing_capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
        "capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
        "dual_sourcing_capped_dual_index_delta_targets": "dual_sourcing_capped_dual_index_delta_targets",
        "capped_dual_index_delta_targets": "dual_sourcing_capped_dual_index_delta_targets",
        "dual_sourcing_capped_dual_index_delta_smallcap_targets": "dual_sourcing_capped_dual_index_delta_smallcap_targets",
        "capped_dual_index_delta_smallcap_targets": "dual_sourcing_capped_dual_index_delta_smallcap_targets",
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
    if normalized == "dual_sourcing_dual_index_delta_targets":
        small_target_upper = min(int(target_upper), int(expedited_max_order_size))
        return {
            "action_dim": 2,
            "action_mode": "discrete_grid",
            "min_values": [0, 0],
            "max_values": [int(small_target_upper), int(regular_max_order_size)],
            "allowed_values": [
                list(range(int(small_target_upper) + 1)),
                list(range(int(regular_max_order_size) + 1)),
            ],
        }
    if normalized == "dual_sourcing_capped_dual_index_targets":
        return {
            "action_dim": 3,
            "action_mode": "vector_quantity",
            "min_values": [0, 0, 0],
            "max_values": [int(target_upper), int(target_upper), int(regular_max_order_size)],
            "allowed_values": None,
        }
    if normalized == "dual_sourcing_capped_dual_index_delta_targets":
        return {
            "action_dim": 3,
            "action_mode": "vector_quantity",
            "min_values": [0, 0, 0],
            "max_values": [
                int(target_upper),
                int(regular_max_order_size),
                int(regular_max_order_size),
            ],
            "allowed_values": None,
        }
    if normalized == "dual_sourcing_capped_dual_index_delta_smallcap_targets":
        small_target_upper = min(int(target_upper), int(expedited_max_order_size))
        small_cap_values = sorted(
            {
                value
                for value in (1, 2, 3, 4, 6, 8, int(regular_max_order_size))
                if 1 <= int(value) <= int(regular_max_order_size)
            }
        )
        return {
            "action_dim": 3,
            "action_mode": "discrete_grid",
            "min_values": [0, 0, small_cap_values[0]],
            "max_values": [int(small_target_upper), int(regular_max_order_size), small_cap_values[-1]],
            "allowed_values": [
                list(range(int(small_target_upper) + 1)),
                list(range(int(regular_max_order_size) + 1)),
                small_cap_values,
            ],
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
