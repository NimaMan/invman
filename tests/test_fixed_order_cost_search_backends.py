import pytest

import invman_rust


def _common_search_kwargs():
    return {
        "current_inventory": 0,
        "lead_time_orders": [0, 0, 0],
        "demands": [5, 2, 7, 4, 3, 9, 2, 1, 6, 5],
        "max_order_size": 8,
        "position_upper_bound": 8,
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "procurement_cost": 0.0,
        "fixed_order_cost": 5.0,
        "warm_up_periods_ratio": 0.2,
        "top_k": 4,
    }


def _direct_cost(policy_name, params, kwargs):
    return invman_rust.lost_sales_fixed_policy_rollout_from_demands(
        policy_name=policy_name,
        params=list(params),
        current_inventory=kwargs["current_inventory"],
        lead_time_orders=kwargs["lead_time_orders"],
        demands=kwargs["demands"],
        max_order_size=kwargs["max_order_size"],
        holding_cost=kwargs["holding_cost"],
        shortage_cost=kwargs["shortage_cost"],
        procurement_cost=kwargs["procurement_cost"],
        fixed_order_cost=kwargs["fixed_order_cost"],
        warm_up_periods_ratio=kwargs["warm_up_periods_ratio"],
    )


def test_rust_s_s_search_returns_sorted_directly_evaluable_results():
    kwargs = _common_search_kwargs()
    best, top = invman_rust.lost_sales_fixed_s_s_search_from_demands(**kwargs)

    assert best == top[0]
    assert len(top) == kwargs["top_k"]
    assert [row[2] for row in top] == sorted(row[2] for row in top)
    assert best[2] == pytest.approx(_direct_cost("s_s", best[:2], kwargs))


def test_rust_s_nq_search_returns_sorted_directly_evaluable_results():
    kwargs = _common_search_kwargs()
    best, top = invman_rust.lost_sales_fixed_s_nq_search_from_demands(**kwargs)

    assert best == top[0]
    assert len(top) == kwargs["top_k"]
    assert [row[2] for row in top] == sorted(row[2] for row in top)
    assert best[2] == pytest.approx(_direct_cost("s_nq", best[:2], kwargs))


def test_rust_modified_s_s_q_search_returns_sorted_directly_evaluable_results():
    kwargs = _common_search_kwargs()
    best, top, evaluated = invman_rust.lost_sales_fixed_modified_s_s_q_search_from_demands(
        **kwargs
    )

    assert best == top[0]
    assert len(top) == kwargs["top_k"]
    assert evaluated > len(top)
    assert [row[3] for row in top] == sorted(row[3] for row in top)
    assert best[3] == pytest.approx(_direct_cost("modified_s_s_q", best[:3], kwargs))
