import pytest

from invman.problems.multi_echelon import (
    build_reference_args,
    search_best_constant_base_stock_policy,
)
from invman.problems.multi_echelon.env import MultiEchelonEnv

invman_rust = pytest.importorskip("invman_rust")


def test_multi_echelon_env_step_respects_action_grid():
    env = MultiEchelonEnv(
        warehouse_lead_time=2,
        retailer_lead_time=2,
        num_retailers=2,
        warehouse_base_stock_levels=[50, 60],
        retailer_base_stock_levels=[0, 5, 10],
        horizon=1,
        track_demand=True,
        warm_up_periods_ratio=0.0,
    )
    env.warehouse_inventory = 20
    env.warehouse_pipeline = [5, 7]
    env.retailer_inventory[:] = [2, 3]
    env.retailer_pipeline[:] = [[1, 0], [0, 2]]
    env.horizon_demands = [[4, 6]]
    env.horizon_expedite_uniforms = [[[0.1] * 20, [0.9] * 20]]

    _, epoch_cost, done = env.step((60, 10))

    assert env.warehouse_pipeline[-1] <= env.warehouse_capacity
    assert env.retailer_pipeline.shape == (2, 2)
    assert epoch_cost >= 0.0
    assert done is True


def test_multi_echelon_constant_base_stock_search_backends_match():
    args = build_reference_args("multi_echelon_setting1")
    args.horizon = 200
    args.warm_up_periods_ratio = 0.0
    args.num_retailers = 3
    args.warehouse_base_stock_levels = [50, 60]
    args.retailer_base_stock_levels = [0, 5, 10]

    python_result = search_best_constant_base_stock_policy(args, seed=222, horizon=200, backend="python")
    rust_result = search_best_constant_base_stock_policy(args, seed=222, horizon=200, backend="rust")
    assert rust_result["best_result"]["params"] == python_result["best_result"]["params"]
    assert rust_result["best_result"]["mean_cost"] == pytest.approx(python_result["best_result"]["mean_cost"])
