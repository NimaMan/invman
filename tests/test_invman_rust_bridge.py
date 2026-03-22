from collections import deque

import numpy as np
import pytest
import torch

from invman.env.lost_sales import LostSalesEnv
from invman.tree_policy import SoftTreePolicy

invman_rust = pytest.importorskip("invman_rust")


def test_rust_soft_tree_action_matches_python():
    torch.manual_seed(7)
    model = SoftTreePolicy(input_dim=4, max_order_size=20, depth=3, temperature=0.25)

    for state_values in (
        [0.1, 0.2, 0.0, 0.5],
        [0.8, 0.1, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ):
        state = torch.tensor(np.array(state_values, dtype=np.float32))
        python_action = int(model(state))
        rust_action = invman_rust.soft_tree_action(
            state=state.tolist(),
            split_weights=model.split_weights.detach().cpu().numpy().reshape(-1).tolist(),
            split_bias=model.split_bias.detach().cpu().numpy().tolist(),
            leaf_logits=model.leaf_logits.detach().cpu().numpy().tolist(),
            depth=model.depth,
            max_order_size=model.max_order_size,
            temperature=model.temperature,
        )
        assert rust_action == python_action


def test_rust_soft_tree_rollout_matches_python_on_fixed_path():
    torch.manual_seed(11)
    model = SoftTreePolicy(input_dim=4, max_order_size=20, depth=3, temperature=0.25)
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    env = LostSalesEnv(
        demand_rate=5.0,
        lead_time=4,
        max_order_size=20,
        horizon=len(demands),
        holding_cost=1.0,
        shortage_cost=4.0,
        track_demand=True,
        warm_up_periods_ratio=0.0,
        state_features="pipeline",
    )
    env.lead_time_orders = deque(lead_time_orders, maxlen=4)
    env.current_inventory_level = current_inventory
    env.current_epoch = 0
    env.done = False
    env.epoch_costs = []
    env.total_cost = 0.0
    env.horizon_demand = np.array(demands, dtype=np.int64)

    state = env.policy_state
    done = False
    while not done:
        action = int(model(torch.as_tensor(state, dtype=torch.float32)))
        state, _, done = env.step(action)

    rust_cost = invman_rust.lost_sales_soft_tree_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        depth=model.depth,
        max_order_size=model.max_order_size,
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
        temperature=model.temperature,
    )

    assert rust_cost == pytest.approx(env.avg_total_cost)
