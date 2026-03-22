from types import SimpleNamespace

import pytest

from invman.problems.lost_sales_fixed_order_cost.heuristics import (
    build_fixed_demand_path,
    evaluate_policy_cost,
    search_best_modified_s_s_q_policy,
    search_best_s_nq_policy,
    search_best_s_s_policy,
)

invman_rust = pytest.importorskip("invman_rust")


def _build_args():
    return SimpleNamespace(
        demand_rate=5.0,
        lead_time=3,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        demand_dist_name="Poisson",
        max_order_size=12,
        horizon=60,
        eval_horizon=60,
        warm_up_periods_ratio=0.2,
        seed=1234,
        track_demand=True,
        state_features="pipeline",
    )


def test_rust_fixed_policy_rollout_matches_python_on_fixed_path():
    args = _build_args()
    fixed_path = build_fixed_demand_path(args=args, seed=77, horizon=40)

    for policy_name, params in (
        ("s_s", {"s": 7, "S": 10}),
        ("s_nq", {"s": 7, "q": 4}),
        ("modified_s_s_q", {"s": 7, "S": 10, "q": 4}),
    ):
        python_cost = evaluate_policy_cost(
            args=args,
            policy_name=policy_name,
            params=params,
            fixed_path=fixed_path,
        )
        rust_cost = invman_rust.lost_sales_fixed_policy_rollout_from_demands(
            policy_name=policy_name,
            params=list(params.values()),
            current_inventory=fixed_path.current_inventory,
            lead_time_orders=list(fixed_path.lead_time_orders),
            demands=list(fixed_path.demands),
            max_order_size=args.max_order_size,
            holding_cost=args.holding_cost,
            shortage_cost=args.shortage_cost,
            procurement_cost=args.procurement_cost,
            fixed_order_cost=args.fixed_order_cost,
            warm_up_periods_ratio=args.warm_up_periods_ratio,
        )
        assert rust_cost == pytest.approx(python_cost)


def test_rust_exhaustive_search_matches_python_on_bounded_instance():
    args = _build_args()
    position_upper_bound = 8

    python_s_s = search_best_s_s_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        top_k=5,
        backend="python",
    )
    rust_s_s = search_best_s_s_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        top_k=5,
        backend="rust",
    )
    assert rust_s_s.best_result.params == python_s_s.best_result.params
    assert rust_s_s.best_result.mean_cost == pytest.approx(python_s_s.best_result.mean_cost)

    python_s_nq = search_best_s_nq_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        top_k=5,
        backend="python",
    )
    rust_s_nq = search_best_s_nq_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        top_k=5,
        backend="rust",
    )
    assert rust_s_nq.best_result.params == python_s_nq.best_result.params
    assert rust_s_nq.best_result.mean_cost == pytest.approx(python_s_nq.best_result.mean_cost)

    python_modified = search_best_modified_s_s_q_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        search_mode="exhaustive",
        backend="python",
    )
    rust_modified = search_best_modified_s_s_q_policy(
        args=args,
        seed=91,
        horizon=50,
        position_upper_bound=position_upper_bound,
        search_mode="exhaustive",
        backend="rust",
    )
    assert rust_modified["modified_policy"].best_result.params == python_modified["modified_policy"].best_result.params
    assert rust_modified["modified_policy"].best_result.mean_cost == pytest.approx(
        python_modified["modified_policy"].best_result.mean_cost
    )
