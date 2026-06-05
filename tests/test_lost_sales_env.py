import math

import pytest

import invman_rust


def _pipeline_state(current_inventory, lead_time_orders):
    state = [float(order) for order in lead_time_orders]
    state[0] += float(max(current_inventory, 0))
    return state


def _linear_categorical_action(flat_params, state, input_dim, output_dim):
    weights = flat_params[: output_dim * input_dim]
    bias = flat_params[output_dim * input_dim :]
    logits = []
    for out_idx in range(output_dim):
        row = weights[out_idx * input_dim : (out_idx + 1) * input_dim]
        logits.append(bias[out_idx] + sum(w * x for w, x in zip(row, state)))
    return max(range(output_dim), key=lambda idx: logits[idx])


def _constant_action_params(input_dim, output_dim, action):
    assert 0 <= action < output_dim
    weights = [0.0] * (input_dim * output_dim)
    bias = [-1.0] * output_dim
    bias[action] = 1.0
    return weights + bias


def _first_slot_threshold_params(input_dim, threshold):
    weights = [0.0] * (input_dim * 2)
    weights[input_dim] = 1.0
    return weights + [0.0, -float(threshold)]


def _mean_after_warmup(epoch_costs, warm_up_periods_ratio):
    warmup = min(math.floor(warm_up_periods_ratio * len(epoch_costs)), len(epoch_costs))
    active = epoch_costs[warmup:] if warmup < len(epoch_costs) else epoch_costs
    return sum(active) / len(active)


def _python_linear_rollout_from_demands(
    *,
    flat_params,
    input_dim,
    output_dim,
    current_inventory,
    lead_time_orders,
    demands,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.0,
    state_normalizer="identity",
    state_scale=None,
):
    current_inventory = int(current_inventory)
    lead_time_orders = [int(order) for order in lead_time_orders]
    epoch_costs = []

    for demand in demands:
        state = _pipeline_state(current_inventory, lead_time_orders)
        if state_normalizer == "quantity_scale":
            if state_scale is None:
                raise ValueError("quantity_scale requires state_scale")
            state = [value / state_scale for value in state]
        action = _linear_categorical_action(flat_params, state, input_dim, output_dim)

        arriving_order = lead_time_orders.pop(0)
        lead_time_orders.append(action)
        current_inventory += arriving_order

        cost = procurement_cost * action
        if action > 0:
            cost += fixed_order_cost
        if demand < current_inventory:
            current_inventory -= demand
            cost += holding_cost * current_inventory
        else:
            lost_sales = demand - current_inventory
            current_inventory = 0
            cost += shortage_cost * lost_sales
        epoch_costs.append(float(cost))

    return _mean_after_warmup(epoch_costs, warm_up_periods_ratio)


def _python_linear_trace_from_demands(
    *,
    flat_params,
    input_dim,
    output_dim,
    current_inventory,
    lead_time_orders,
    demands,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.0,
    state_normalizer="identity",
    state_scale=None,
):
    current_inventory = int(current_inventory)
    lead_time_orders = [int(order) for order in lead_time_orders]
    epoch_costs = []
    trace = []
    warmup = min(math.floor(warm_up_periods_ratio * len(demands)), len(demands))

    for period, demand in enumerate(demands):
        current_inventory_before_order = current_inventory
        pipeline_before_order = list(lead_time_orders)
        raw_state = _pipeline_state(current_inventory, lead_time_orders)
        normalized_state = raw_state
        if state_normalizer == "quantity_scale":
            if state_scale is None:
                raise ValueError("quantity_scale requires state_scale")
            normalized_state = [value / state_scale for value in raw_state]
        action = _linear_categorical_action(
            flat_params,
            normalized_state,
            input_dim,
            output_dim,
        )

        arriving_order = lead_time_orders.pop(0)
        lead_time_orders.append(action)
        current_inventory += arriving_order
        inventory_before_demand = current_inventory

        cost = procurement_cost * action
        if action > 0:
            cost += fixed_order_cost
        if demand < current_inventory:
            current_inventory -= demand
            cost += holding_cost * current_inventory
        else:
            lost_sales = demand - current_inventory
            current_inventory = 0
            cost += shortage_cost * lost_sales
        epoch_costs.append(float(cost))
        trace.append(
            {
                "period": period,
                "demand": demand,
                "current_inventory_before_order": current_inventory_before_order,
                "pipeline_before_order": pipeline_before_order,
                "raw_state": raw_state,
                "normalized_state": normalized_state,
                "order_quantity": action,
                "arriving_order": arriving_order,
                "inventory_before_demand": inventory_before_demand,
                "ending_inventory": current_inventory,
                "period_cost": float(cost),
                "active_after_warmup": period >= warmup,
            }
        )

    return _mean_after_warmup(epoch_costs, warm_up_periods_ratio), trace


def _rust_linear_rollout_from_demands(**kwargs):
    return invman_rust.lost_sales_linear_rollout_from_demands(
        policy_head="categorical_quantity",
        policy_max_quantity=None,
        **kwargs,
    )


def _rust_linear_trace_from_demands(**kwargs):
    return invman_rust.lost_sales_linear_trace_from_demands(
        policy_head="categorical_quantity",
        policy_max_quantity=None,
        **kwargs,
    )


def test_linear_rollout_adds_current_inventory_to_first_pipeline_slot_before_action():
    params = _first_slot_threshold_params(input_dim=3, threshold=9.0)
    kwargs = {
        "flat_params": params,
        "input_dim": 3,
        "output_dim": 2,
        "current_inventory": 6,
        "lead_time_orders": [4, 2, 1],
        "demands": [0],
        "holding_cost": 0.0,
        "shortage_cost": 0.0,
        "procurement_cost": 5.0,
        "fixed_order_cost": 7.0,
        "warm_up_periods_ratio": 0.0,
    }

    actual = _rust_linear_rollout_from_demands(**kwargs)
    expected = _python_linear_rollout_from_demands(**kwargs)

    assert expected == 12.0
    assert actual == pytest.approx(expected)


def test_linear_rollout_applies_holding_cost_after_arrival_and_demand():
    params = _constant_action_params(input_dim=2, output_dim=1, action=0)
    kwargs = {
        "flat_params": params,
        "input_dim": 2,
        "output_dim": 1,
        "current_inventory": 5,
        "lead_time_orders": [0, 0],
        "demands": [3],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 0.0,
        "warm_up_periods_ratio": 0.0,
    }

    assert _rust_linear_rollout_from_demands(**kwargs) == pytest.approx(
        _python_linear_rollout_from_demands(**kwargs)
    )
    assert _python_linear_rollout_from_demands(**kwargs) == 2.0


def test_linear_rollout_applies_shortage_procurement_and_fixed_costs():
    params = _constant_action_params(input_dim=2, output_dim=4, action=3)
    kwargs = {
        "flat_params": params,
        "input_dim": 2,
        "output_dim": 4,
        "current_inventory": 1,
        "lead_time_orders": [0, 0],
        "demands": [4],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 2.0,
        "fixed_order_cost": 7.0,
        "warm_up_periods_ratio": 0.0,
    }

    assert _rust_linear_rollout_from_demands(**kwargs) == pytest.approx(
        _python_linear_rollout_from_demands(**kwargs)
    )
    assert _python_linear_rollout_from_demands(**kwargs) == 25.0


def test_linear_rollout_matches_python_oracle_on_state_dependent_multi_period_path():
    params = _first_slot_threshold_params(input_dim=2, threshold=2.5)
    kwargs = {
        "flat_params": params,
        "input_dim": 2,
        "output_dim": 2,
        "current_inventory": 2,
        "lead_time_orders": [1, 0],
        "demands": [0, 5, 0, 2],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.5,
        "fixed_order_cost": 3.0,
        "warm_up_periods_ratio": 0.0,
    }

    assert _rust_linear_rollout_from_demands(**kwargs) == pytest.approx(
        _python_linear_rollout_from_demands(**kwargs)
    )


def test_linear_trace_matches_python_oracle_on_state_dependent_multi_period_path():
    params = _first_slot_threshold_params(input_dim=2, threshold=2.5)
    kwargs = {
        "flat_params": params,
        "input_dim": 2,
        "output_dim": 2,
        "current_inventory": 2,
        "lead_time_orders": [1, 0],
        "demands": [0, 5, 0, 2],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.5,
        "fixed_order_cost": 3.0,
        "warm_up_periods_ratio": 0.25,
    }

    actual = _rust_linear_trace_from_demands(**kwargs)
    expected_mean, expected_trace = _python_linear_trace_from_demands(**kwargs)

    assert actual["policy_name"] == "linear"
    assert actual["mean_cost"] == pytest.approx(expected_mean)
    assert actual["warm_up_periods"] == 1
    assert len(actual["trace"]) == len(expected_trace)
    exact_fields = [
        "period",
        "demand",
        "current_inventory_before_order",
        "pipeline_before_order",
        "order_quantity",
        "arriving_order",
        "inventory_before_demand",
        "ending_inventory",
        "active_after_warmup",
    ]
    for actual_row, expected_row in zip(actual["trace"], expected_trace):
        for field in exact_fields:
            assert actual_row[field] == expected_row[field]
        assert actual_row["raw_state"] == pytest.approx(expected_row["raw_state"])
        assert actual_row["normalized_state"] == pytest.approx(
            expected_row["normalized_state"]
        )
        assert actual_row["period_cost"] == pytest.approx(expected_row["period_cost"])

    scalar = _rust_linear_rollout_from_demands(**kwargs)
    assert actual["mean_cost"] == pytest.approx(scalar)


def test_soft_tree_and_nn_trace_bindings_match_scalar_rollouts():
    common = {
        "input_dim": 2,
        "current_inventory": 0,
        "lead_time_orders": [0, 0],
        "demands": [1, 0, 2],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 0.0,
        "warm_up_periods_ratio": 0.0,
    }
    soft_tree_params = [0.0] * 9
    soft_tree_trace = invman_rust.lost_sales_soft_tree_trace_from_demands(
        flat_params=soft_tree_params,
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        policy_max_quantity=None,
        **common,
    )
    soft_tree_scalar = invman_rust.lost_sales_soft_tree_rollout_from_demands(
        flat_params=soft_tree_params,
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        policy_max_quantity=None,
        **common,
    )
    assert soft_tree_trace["policy_name"] == "soft_tree"
    assert soft_tree_trace["mean_cost"] == pytest.approx(soft_tree_scalar)
    assert len(soft_tree_trace["trace"]) == len(common["demands"])

    nn_params = [0.0] * 9
    nn_trace = invman_rust.lost_sales_nn_trace_from_demands(
        flat_params=nn_params,
        hidden_dims=[2],
        output_dim=1,
        policy_max_quantity=None,
        activation="relu",
        policy_head="direct_quantity",
        **common,
    )
    nn_scalar = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=nn_params,
        hidden_dims=[2],
        output_dim=1,
        policy_max_quantity=None,
        activation="relu",
        policy_head="direct_quantity",
        **common,
    )
    assert nn_trace["policy_name"] == "nn"
    assert nn_trace["mean_cost"] == pytest.approx(nn_scalar)
    assert len(nn_trace["trace"]) == len(common["demands"])


def test_linear_rollout_warmup_uses_floor_and_full_path_when_all_warmup():
    params = _constant_action_params(input_dim=1, output_dim=1, action=0)
    kwargs = {
        "flat_params": params,
        "input_dim": 1,
        "output_dim": 1,
        "current_inventory": 0,
        "lead_time_orders": [0],
        "demands": [5, 0, 3, 0],
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 0.0,
    }

    half_warmup = {**kwargs, "warm_up_periods_ratio": 0.5}
    all_warmup = {**kwargs, "warm_up_periods_ratio": 1.0}

    assert _rust_linear_rollout_from_demands(**half_warmup) == pytest.approx(
        _python_linear_rollout_from_demands(**half_warmup)
    )
    assert _python_linear_rollout_from_demands(**half_warmup) == 6.0

    assert _rust_linear_rollout_from_demands(**all_warmup) == pytest.approx(
        _python_linear_rollout_from_demands(**all_warmup)
    )
    assert _python_linear_rollout_from_demands(**all_warmup) == 8.0


def test_linear_rollout_quantity_scale_normalizes_pipeline_state_for_action_selection():
    params = _first_slot_threshold_params(input_dim=3, threshold=0.49)
    kwargs = {
        "flat_params": params,
        "input_dim": 3,
        "output_dim": 2,
        "current_inventory": 6,
        "lead_time_orders": [4, 2, 1],
        "demands": [0],
        "holding_cost": 0.0,
        "shortage_cost": 0.0,
        "procurement_cost": 5.0,
        "fixed_order_cost": 7.0,
        "warm_up_periods_ratio": 0.0,
        "state_normalizer": "quantity_scale",
        "state_scale": 20.0,
    }

    actual = _rust_linear_rollout_from_demands(**kwargs)
    expected = _python_linear_rollout_from_demands(**kwargs)

    assert expected == 12.0
    assert actual == pytest.approx(expected)


def test_linear_rollout_validates_state_shape_and_normalizer_arguments():
    params = _constant_action_params(input_dim=3, output_dim=1, action=0)

    with pytest.raises(ValueError, match="input_dim must match lead_time"):
        _rust_linear_rollout_from_demands(
            flat_params=params,
            input_dim=3,
            output_dim=1,
            current_inventory=0,
            lead_time_orders=[0, 0],
            demands=[1],
            warm_up_periods_ratio=0.0,
        )

    with pytest.raises(ValueError, match="requires state_scale"):
        _rust_linear_rollout_from_demands(
            flat_params=params,
            input_dim=3,
            output_dim=1,
            current_inventory=0,
            lead_time_orders=[0, 0, 0],
            demands=[1],
            warm_up_periods_ratio=0.0,
            state_normalizer="quantity_scale",
        )


def test_constant_action_rollout_is_seed_deterministic_for_mmpp2_demand():
    kwargs = {
        "demand_rate": 5.0,
        "demand_dist_name": "MarkovModulatedPoisson2",
        "demand_lambda_low": 3.0,
        "demand_lambda_high": 7.0,
        "demand_p00": 0.9,
        "demand_p11": 0.9,
        "lead_time": 4,
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "horizon": 100,
        "action": 5,
        "seed": 123,
        "warm_up_periods_ratio": 0.2,
    }

    first = invman_rust.lost_sales_constant_action_rollout(**kwargs)
    second = invman_rust.lost_sales_constant_action_rollout(**kwargs)

    assert math.isfinite(first)
    assert first == pytest.approx(second)


def test_constant_action_rollout_validates_lead_time_and_warmup():
    with pytest.raises(ValueError, match="lead_time must be at least 1"):
        invman_rust.lost_sales_constant_action_rollout(
            demand_rate=5.0,
            lead_time=0,
        )

    with pytest.raises(ValueError, match="warm_up_periods_ratio must be in"):
        invman_rust.lost_sales_constant_action_rollout(
            demand_rate=5.0,
            warm_up_periods_ratio=1.1,
        )
