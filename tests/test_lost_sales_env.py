from collections import deque

import numpy as np

from invman.env.lost_sales import LostSalesEnv


def test_reset_reinitializes_episode_state():
    env = LostSalesEnv(
        demand_rate=5.0,
        lead_time=2,
        max_order_size=10,
        horizon=20,
        track_demand=True,
        warm_up_periods_ratio=0.0,
    )
    env.current_epoch = 7
    env.done = True
    env.epoch_costs = [1.0, 2.0]

    state = env.reset()

    assert env.current_epoch == 0
    assert env.done is False
    assert env.epoch_costs == []
    assert len(env.lead_time_orders) == env.lead_time
    assert state.shape == (env.state_space_dim,)


def test_step_applies_holding_cost_correctly():
    env = LostSalesEnv(
        demand_rate=5.0,
        lead_time=2,
        max_order_size=10,
        horizon=3,
        holding_cost=1.0,
        shortage_cost=4.0,
        track_demand=True,
        warm_up_periods_ratio=0.0,
    )
    env.lead_time_orders = deque([0, 0], maxlen=2)
    env.current_inventory_level = 5
    env.current_epoch = 0
    env.done = False
    env.horizon_demand = np.array([3, 0, 0])
    env.epoch_costs = []

    _, cost, done = env.step(0)

    assert cost == 2.0
    assert env.current_inventory_level == 2
    assert done is False


def test_step_applies_shortage_procurement_and_fixed_costs():
    env = LostSalesEnv(
        demand_rate=5.0,
        lead_time=2,
        max_order_size=10,
        horizon=3,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=2.0,
        fixed_order_cost=7.0,
        track_demand=True,
        warm_up_periods_ratio=0.0,
    )
    env.lead_time_orders = deque([0, 0], maxlen=2)
    env.current_inventory_level = 1
    env.current_epoch = 0
    env.done = False
    env.horizon_demand = np.array([4, 0, 0])
    env.epoch_costs = []

    _, cost, _ = env.step(3)

    assert cost == 25.0
    assert env.current_inventory_level == 0


def test_track_demand_makes_demand_path_independent_of_action_cap():
    np.random.seed(123)
    env_small_cap = LostSalesEnv(
        demand_rate=5.0,
        lead_time=4,
        max_order_size=20,
        horizon=50,
        track_demand=True,
    )
    np.random.seed(123)
    env_large_cap = LostSalesEnv(
        demand_rate=5.0,
        lead_time=4,
        max_order_size=40,
        horizon=50,
        track_demand=True,
    )

    assert np.array_equal(env_small_cap.horizon_demand, env_large_cap.horizon_demand)
