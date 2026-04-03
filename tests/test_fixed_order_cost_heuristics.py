from types import SimpleNamespace

import pytest

from invman.problems.lost_sales_fixed_order_cost.heuristics import (
    get_protection_period_demand_variance,
    get_modified_s_s_q_order_quantity,
    get_paper_q_heuristic,
    get_review_period_demand_variance,
    get_s_nq_order_quantity,
    get_s_s_order_quantity,
)


def test_s_s_orders_up_to_level_when_below_reorder_point():
    assert get_s_s_order_quantity(inventory_position=8, s=10, S=15, max_order_size=50) == 7
    assert get_s_s_order_quantity(inventory_position=12, s=10, S=15, max_order_size=50) == 0


def test_s_nq_orders_minimum_multiple_that_exceeds_reorder_point():
    assert get_s_nq_order_quantity(inventory_position=7, s=10, q=4, max_order_size=50) == 4
    assert get_s_nq_order_quantity(inventory_position=4, s=10, q=4, max_order_size=50) == 8
    assert get_s_nq_order_quantity(inventory_position=11, s=10, q=4, max_order_size=50) == 0


def test_modified_s_s_q_respects_upper_bound():
    assert get_modified_s_s_q_order_quantity(inventory_position=6, s=10, S=18, q=5, max_order_size=50) == 5
    assert get_modified_s_s_q_order_quantity(inventory_position=9, s=10, S=12, q=5, max_order_size=50) == 3
    assert get_modified_s_s_q_order_quantity(inventory_position=11, s=10, S=18, q=5, max_order_size=50) == 0


def test_paper_q_heuristic_stays_within_policy_bounds():
    args = SimpleNamespace(demand_rate=5.0, demand_dist_name="Poisson", lead_time=4, max_order_size=50)
    q = get_paper_q_heuristic(args=args, s=18, S=24)
    assert 6 <= q <= 24


def test_markov_modulated_variance_reflects_correlation_sign():
    positive_args = SimpleNamespace(
        demand_rate=5.0,
        demand_dist_name="MarkovModulatedPoisson2",
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
        lead_time=4,
        max_order_size=50,
    )
    negative_args = SimpleNamespace(
        demand_rate=5.0,
        demand_dist_name="MarkovModulatedPoisson2",
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.1,
        demand_p11=0.1,
        lead_time=4,
        max_order_size=50,
    )

    one_period_positive = get_review_period_demand_variance(
        positive_args.demand_dist_name,
        positive_args.demand_rate,
        demand_lambda_low=positive_args.demand_lambda_low,
        demand_lambda_high=positive_args.demand_lambda_high,
        demand_p00=positive_args.demand_p00,
        demand_p11=positive_args.demand_p11,
    )
    one_period_negative = get_review_period_demand_variance(
        negative_args.demand_dist_name,
        negative_args.demand_rate,
        demand_lambda_low=negative_args.demand_lambda_low,
        demand_lambda_high=negative_args.demand_lambda_high,
        demand_p00=negative_args.demand_p00,
        demand_p11=negative_args.demand_p11,
    )
    assert one_period_positive == pytest.approx(9.0)
    assert one_period_negative == pytest.approx(9.0)

    positive_protection = get_protection_period_demand_variance(positive_args, 5)
    negative_protection = get_protection_period_demand_variance(negative_args, 5)
    assert positive_protection > negative_protection
