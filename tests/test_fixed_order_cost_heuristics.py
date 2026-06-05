import math
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import invman_rust

from invman.policy import Policy
from scripts.lost_sales_fixed_order_cost.diagnostics.analyze_policy import _rust_policy_trace


def _policy_action(policy_name, params, inventory_position, max_order_size):
    if policy_name == "s_s":
        s, s_up_to = params
        if inventory_position > s:
            return 0
        return min(max(s_up_to - max(inventory_position, 0), 0), max_order_size)
    if policy_name == "s_nq":
        s, q = params
        if inventory_position > s:
            return 0
        deficit = max(s + 1 - max(inventory_position, 0), 0)
        return min(math.ceil(deficit / q) * q, max_order_size)
    if policy_name == "modified_s_s_q":
        s, s_up_to, q = params
        if inventory_position > s:
            return 0
        return min(q, max(s_up_to - max(inventory_position, 0), 0), max_order_size)
    raise ValueError(policy_name)


def _python_fixed_policy_rollout(
    *,
    policy_name,
    params,
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=5.0,
    warm_up_periods_ratio=0.2,
):
    current_inventory = int(current_inventory)
    lead_time_orders = list(lead_time_orders)
    epoch_costs = []
    for demand in demands:
        inventory_position = current_inventory + sum(lead_time_orders)
        action = _policy_action(policy_name, params, inventory_position, max_order_size)
        arriving = lead_time_orders.pop(0)
        lead_time_orders.append(action)
        current_inventory += arriving

        cost = procurement_cost * action
        if action > 0:
            cost += fixed_order_cost
        if demand < current_inventory:
            current_inventory -= demand
            cost += current_inventory * holding_cost
        else:
            lost_sales = demand - current_inventory
            current_inventory = 0
            cost += shortage_cost * lost_sales
        epoch_costs.append(float(cost))

    warmup = min(int(math.floor(warm_up_periods_ratio * len(epoch_costs))), len(epoch_costs))
    active_costs = epoch_costs[warmup:] if warmup < len(epoch_costs) else epoch_costs
    return sum(active_costs) / len(active_costs)


def _python_fixed_policy_trace(
    *,
    policy_name,
    params,
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=5.0,
    warm_up_periods_ratio=0.2,
):
    current_inventory = int(current_inventory)
    lead_time_orders = list(lead_time_orders)
    epoch_costs = []
    trace = []
    warmup = min(int(math.floor(warm_up_periods_ratio * len(demands))), len(demands))
    for period, demand in enumerate(demands):
        pipeline_before_order = list(lead_time_orders)
        inventory_position = current_inventory + sum(lead_time_orders)
        action = _policy_action(policy_name, params, inventory_position, max_order_size)
        arriving = lead_time_orders.pop(0)
        lead_time_orders.append(action)
        current_inventory += arriving
        inventory_before_demand = current_inventory

        cost = procurement_cost * action
        if action > 0:
            cost += fixed_order_cost
        if demand < current_inventory:
            current_inventory -= demand
            cost += current_inventory * holding_cost
        else:
            lost_sales = demand - current_inventory
            current_inventory = 0
            cost += shortage_cost * lost_sales
        epoch_costs.append(float(cost))
        trace.append(
            {
                "period": period,
                "demand": demand,
                "pipeline_before_order": pipeline_before_order,
                "inventory_position_before_order": inventory_position,
                "order_quantity": action,
                "arriving_order": arriving,
                "inventory_before_demand": inventory_before_demand,
                "ending_inventory": current_inventory,
                "period_cost": float(cost),
                "active_after_warmup": period >= warmup,
            }
        )
    active_costs = epoch_costs[warmup:] if warmup < len(epoch_costs) else epoch_costs
    return sum(active_costs) / len(active_costs), trace


def test_fixed_cost_policy_rollout_matches_independent_oracle_on_fixed_path():
    common = {
        "current_inventory": 4,
        "lead_time_orders": [1, 2, 0],
        "demands": [3, 8, 1, 5, 2, 0, 7, 4, 6, 3],
        "max_order_size": 12,
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 5.0,
        "warm_up_periods_ratio": 0.2,
    }

    for policy_name, params in (
        ("s_s", [7, 10]),
        ("s_nq", [7, 4]),
        ("modified_s_s_q", [7, 10, 4]),
    ):
        expected = _python_fixed_policy_rollout(
            policy_name=policy_name,
            params=params,
            **common,
        )
        actual = invman_rust.lost_sales_fixed_policy_rollout_from_demands(
            policy_name=policy_name,
            params=params,
            **common,
        )
        assert actual == pytest.approx(expected)


def test_fixed_cost_policy_trace_matches_independent_oracle_on_fixed_path():
    common = {
        "current_inventory": 4,
        "lead_time_orders": [1, 2, 0],
        "demands": [3, 8, 1, 5, 2, 0, 7, 4, 6, 3],
        "max_order_size": 12,
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 5.0,
        "warm_up_periods_ratio": 0.2,
    }
    expected_mean, expected_trace = _python_fixed_policy_trace(
        policy_name="modified_s_s_q",
        params=[7, 10, 4],
        **common,
    )

    actual = invman_rust.lost_sales_fixed_policy_trace_from_demands(
        policy_name="modified_s_s_q",
        params=[7, 10, 4],
        **common,
    )

    assert actual["policy_name"] == "modified_s_s_q"
    assert actual["params"] == [7, 10, 4]
    assert actual["mean_cost"] == pytest.approx(expected_mean)
    assert actual["warm_up_periods"] == 2
    assert len(actual["trace"]) == len(expected_trace)
    exact_fields = [
        "period",
        "demand",
        "pipeline_before_order",
        "inventory_position_before_order",
        "order_quantity",
        "arriving_order",
        "inventory_before_demand",
        "ending_inventory",
        "active_after_warmup",
    ]
    for actual_row, expected_row in zip(actual["trace"], expected_trace):
        for field in exact_fields:
            assert actual_row[field] == expected_row[field]
        assert actual_row["period_cost"] == pytest.approx(expected_row["period_cost"])
    assert [row["active_after_warmup"] for row in actual["trace"]] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    scalar = invman_rust.lost_sales_fixed_policy_rollout_from_demands(
        policy_name="modified_s_s_q",
        params=[7, 10, 4],
        **common,
    )
    assert actual["mean_cost"] == pytest.approx(scalar)


def test_fixed_cost_diagnostic_learned_policy_trace_uses_rust_binding():
    policy = Policy(
        backbone="linear",
        input_dim=2,
        output_dim=1,
        action_output_mode="categorical_quantity",
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(5,),
        max_order_size=5,
    )
    args = SimpleNamespace(
        demand_rate=3.0,
        lead_time=2,
        horizon=20,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        warm_up_periods_ratio=0.2,
    )

    payload = _rust_policy_trace(policy, args, trace_horizon=5, trace_rows=3)

    assert payload["trace_demands"] == {
        "source": "deterministic_rounded_mean_demand",
        "horizon": 5,
        "value": 3,
    }
    assert payload["trace_summary"]["all_periods"]["unique_actions"] == 1
    assert len(payload["trace_head"]) == 3
    assert payload["trace_head"][0]["order_quantity"] == 0


def test_fixed_cost_diagnostic_cli_emits_learned_and_heuristic_traces(tmp_path):
    policy = Policy(
        backbone="linear",
        input_dim=4,
        output_dim=1,
        action_output_mode="categorical_quantity",
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(50,),
        max_order_size=50,
        state_normalizer="identity",
    )
    model_dir = tmp_path / "policy"
    output_json = tmp_path / "diagnostics" / "policy_trace.json"
    policy.save(model_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/lost_sales_fixed_order_cost/diagnostics/analyze_policy.py",
            "--model_dir",
            str(model_dir),
            "--horizon",
            "5",
            "--trace_horizon",
            "4",
            "--trace_rows",
            "2",
            "--output_json",
            str(output_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)
    archived_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert archived_payload == payload
    model_summary = payload["model_rollout"]["action_summary"]
    heuristic_summary = payload["modified_s_s_q_rollout"]["action_summary"]
    assert model_summary["trace_demands"]["horizon"] == 4
    assert len(model_summary["trace_head"]) == 2
    assert heuristic_summary["params"] is not None
    assert len(heuristic_summary["trace_head"]) == 2
    assert payload["coarse_state_grid"]["unique_actions"] == 1


def test_fixed_cost_heuristics_all_reports_three_rust_baselines():
    summary = invman_rust.lost_sales_fixed_heuristics_all(
        "Poisson",
        5.0,
        3.0,
        7.0,
        0.9,
        0.9,
        4,
        1.0,
        4.0,
        0.0,
        5.0,
        12,
        20,
        80,
        123,
        0.2,
        1,
    )

    assert set(summary) == {"s_s", "s_nq", "modified_s_s_q"}
    assert all(cost > 0.0 for cost in summary.values())
