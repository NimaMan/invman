from types import SimpleNamespace

from invman.problems.lost_sales_fixed_order_cost.heuristics import (
    get_modified_s_s_q_order_quantity,
    get_paper_q_heuristic,
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
