from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from invman.policies import LinearPolicyNet, PolicyNet, SoftTreePolicy
from invman.problems.lost_sales.env import LostSalesEnv, get_population_fitness

invman_rust = pytest.importorskip("invman_rust")


def _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands):
    env = LostSalesEnv(
        demand_rate=float(np.mean(demands)) if len(demands) else 0.0,
        lead_time=len(lead_time_orders),
        max_order_size=model.max_order_size,
        horizon=len(demands),
        holding_cost=1.0,
        shortage_cost=4.0,
        track_demand=True,
        warm_up_periods_ratio=0.0,
        state_features="pipeline",
    )
    env.lead_time_orders = deque(lead_time_orders, maxlen=len(lead_time_orders))
    env.current_inventory_level = current_inventory
    env.current_epoch = 0
    env.done = False
    env.epoch_costs = []
    env.total_cost = 0.0
    env.horizon_demand = np.array(demands, dtype=np.int64)

    state = env.policy_state
    done = False
    while not done:
        action = int(model(state))
        state, _, done = env.step(action)
    return env.avg_total_cost


def _make_rollout_args(*, demand_dist_name="Poisson", rollout_backend="rust"):
    return SimpleNamespace(
        demand_rate=5.0,
        demand_dist_name=demand_dist_name,
        lead_time=4,
        max_order_size=20,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        horizon=40,
        eval_horizon=40,
        warm_up_periods_ratio=0.2,
        rollout_backend=rollout_backend,
        state_features="pipeline",
    )


def test_rust_soft_tree_action_matches_python():
    np.random.seed(7)
    model = SoftTreePolicy(input_dim=4, max_order_size=20, depth=3, temperature=0.25)

    for state_values in (
        [0.1, 0.2, 0.0, 0.5],
        [0.8, 0.1, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ):
        state = np.array(state_values, dtype=np.float32)
        python_action = int(model(state))
        rust_action = invman_rust.soft_tree_action(
            state=state.tolist(),
            split_weights=model.split_weights.reshape(-1).tolist(),
            split_bias=model.split_bias.tolist(),
            leaf_logits=model.leaf_logits.reshape(-1).tolist(),
            depth=model.depth,
            policy_max_quantity=model.max_order_size,
            temperature=model.temperature,
        )
        assert rust_action == python_action


def test_rust_axis_aligned_soft_tree_action_matches_python():
    np.random.seed(19)
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=3,
        temperature=0.25,
        split_type="axis_aligned",
    )

    for state_values in (
        [0.1, 0.2, 0.0, 0.5],
        [0.8, 0.1, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ):
        state = np.array(state_values, dtype=np.float32)
        python_action = int(model(state))
        rust_action = invman_rust.soft_tree_action(
            state=state.tolist(),
            split_weights=model.split_weights.reshape(-1).tolist(),
            split_bias=model.split_bias.tolist(),
            leaf_logits=model.leaf_logits.reshape(-1).tolist(),
            depth=model.depth,
            policy_max_quantity=model.max_order_size,
            temperature=model.temperature,
            split_type="axis_aligned",
        )
        assert rust_action == python_action

def test_rust_linear_leaf_soft_tree_rollout_matches_python_on_fixed_path():
    np.random.seed(23)
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    env = LostSalesEnv(
        demand_rate=float(np.mean(demands)),
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
        action = int(model(state))
        state, _, done = env.step(action)

    rust_cost = invman_rust.lost_sales_soft_tree_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        depth=model.depth,
        split_type="oblique",
        leaf_type="linear",
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


def test_rust_sigmoid_linear_leaf_soft_tree_rollout_matches_python_on_fixed_path():
    np.random.seed(29)
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    env = LostSalesEnv(
        demand_rate=float(np.mean(demands)),
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
        action = int(model(state))
        state, _, done = env.step(action)

    rust_cost = invman_rust.lost_sales_soft_tree_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        depth=model.depth,
        policy_max_quantity=20,
        split_type="oblique",
        leaf_type="sigmoid_linear",
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


def test_rust_linear_leaf_soft_tree_action_matches_python():
    np.random.seed(23)
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    for state_values in (
        [0.1, 0.2, 0.0, 0.5],
        [0.8, 0.1, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ):
        state = np.array(state_values, dtype=np.float32)
        python_action = int(model(state))
        rust_action = invman_rust.soft_tree_action_from_flat_params(
            state=state.tolist(),
            flat_params=flat_params.tolist(),
            input_dim=model.input_dim,
            depth=model.depth,
            temperature=model.temperature,
            split_type="oblique",
            leaf_type="linear",
        )
        assert rust_action == python_action

def test_rust_soft_tree_population_rollout_matches_single_rollouts():
    np.random.seed(13)
    model_a = SoftTreePolicy(input_dim=4, max_order_size=20, depth=3, temperature=0.25, leaf_type="linear")
    np.random.seed(17)
    model_b = SoftTreePolicy(input_dim=4, max_order_size=20, depth=3, temperature=0.25, leaf_type="linear")

    params_batch = [
        model_a.get_model_flat_params().astype(np.float32).tolist(),
        model_b.get_model_flat_params().astype(np.float32).tolist(),
    ]
    seeds = [123, 456]

    batch_costs = invman_rust.lost_sales_soft_tree_population_rollout(
        params_batch=params_batch,
        input_dim=4,
        depth=3,
        split_type="oblique",
        leaf_type="linear",
        demand_rate=5.0,
        seeds=seeds,
        lead_time=4,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        horizon=200,
        warm_up_periods_ratio=0.2,
        temperature=0.25,
    )

    single_costs = [
        invman_rust.lost_sales_soft_tree_rollout(
            flat_params=params_batch[idx],
            input_dim=4,
            depth=3,
            split_type="oblique",
            leaf_type="linear",
            demand_rate=5.0,
            lead_time=4,
            holding_cost=1.0,
            shortage_cost=4.0,
            procurement_cost=0.0,
            fixed_order_cost=0.0,
            horizon=200,
            seed=seeds[idx],
            warm_up_periods_ratio=0.2,
            temperature=0.25,
        )
        for idx in range(len(params_batch))
    ]

    assert batch_costs == pytest.approx(single_costs)


def test_rust_linear_geometric_population_rollout_matches_single_rollouts():
    np.random.seed(43)
    model_a = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )
    np.random.seed(47)
    model_b = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )

    params_batch = [
        model_a.get_model_flat_params().astype(np.float32).tolist(),
        model_b.get_model_flat_params().astype(np.float32).tolist(),
    ]
    seeds = [123, 456]

    batch_costs = invman_rust.lost_sales_linear_population_rollout(
        params_batch=params_batch,
        input_dim=4,
        output_dim=21,
        policy_max_quantity=20,
        demand_rate=5.0,
        demand_dist_name="Geometric",
        seeds=seeds,
        lead_time=4,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        horizon=200,
        warm_up_periods_ratio=0.2,
    )

    single_costs = [
        invman_rust.lost_sales_linear_rollout(
            flat_params=params_batch[idx],
            input_dim=4,
            output_dim=21,
            policy_max_quantity=20,
            demand_rate=5.0,
            demand_dist_name="Geometric",
            lead_time=4,
            holding_cost=1.0,
            shortage_cost=4.0,
            procurement_cost=0.0,
            fixed_order_cost=0.0,
            horizon=200,
            seed=seeds[idx],
            warm_up_periods_ratio=0.2,
        )
        for idx in range(len(params_batch))
    ]

    assert batch_costs == pytest.approx(single_costs)


def test_geometric_population_fitness_uses_rust_backend():
    np.random.seed(53)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )
    args = _make_rollout_args(demand_dist_name="Geometric", rollout_backend="rust")
    params_batch = [
        model.get_model_flat_params().astype(np.float32),
        model.get_model_flat_params().astype(np.float32) + 0.1,
    ]

    fitness = get_population_fitness(model, args, params_batch, seeds=[11, 22])

    assert fitness is not None
    assert len(fitness) == 2
    assert all(isinstance(score, tuple) and len(score) == 2 for score in fitness)


def test_markov_modulated_population_fitness_uses_rust_backend():
    np.random.seed(59)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )
    args = _make_rollout_args(demand_dist_name="MarkovModulatedPoisson2", rollout_backend="rust")
    args.demand_lambda_low = 3.0
    args.demand_lambda_high = 7.0
    args.demand_p00 = 0.9
    args.demand_p11 = 0.9
    params_batch = [
        model.get_model_flat_params().astype(np.float32),
        model.get_model_flat_params().astype(np.float32) + 0.1,
    ]

    fitness = get_population_fitness(model, args, params_batch, seeds=[11, 22])

    assert fitness is not None
    assert len(fitness) == 2
    assert all(isinstance(score, tuple) and len(score) == 2 for score in fitness)


def test_rust_linear_rollout_matches_python_on_fixed_path():
    np.random.seed(29)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        policy_max_quantity=model.max_order_size,
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_linear_gated_ordinal_rollout_matches_python_on_fixed_path():
    np.random.seed(37)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="soft_gated_ordinal_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="soft_gated_ordinal_quantity",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_linear_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(33)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=None,
        policy_head="direct_quantity",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_linear_sigmoid_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(34)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="sigmoid_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="sigmoid_direct_quantity",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_linear_gated_direct_rollout_ignores_policy_cap():
    flat_params = np.array([0.0, 0.0, 10.0, 30.0], dtype=np.float32)

    uncapped_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=1,
        output_dim=2,
        policy_max_quantity=None,
        policy_head="soft_gated_direct_quantity",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )
    capped_argument_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=1,
        output_dim=2,
        policy_max_quantity=20,
        policy_head="soft_gated_direct_quantity",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert uncapped_cost == pytest.approx(capped_argument_cost)
    assert uncapped_cost == pytest.approx(15.0)


def test_rust_linear_gated_sigmoid_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(37)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="gated_sigmoid_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="gated_sigmoid_direct_quantity",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_linear_hard_gated_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(38)
    model = LinearPolicyNet(
        input_dim=4,
        output_dim=21,
        action_output_mode="hard_gated_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="hard_gated_direct_quantity",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_nn_rollout_matches_python_on_fixed_path():
    np.random.seed(31)
    model = PolicyNet(
        input_dim=4,
        hidden_dim=[8],
        output_dim=21,
        activation="selu",
        action_output_mode="categorical_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        hidden_dims=model.hidden_dim,
        output_dim=model.output_dim,
        policy_max_quantity=model.max_order_size,
        activation="selu",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_nn_sigmoid_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(32)
    model = PolicyNet(
        input_dim=4,
        hidden_dim=[8],
        output_dim=21,
        activation="selu",
        action_output_mode="sigmoid_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        hidden_dims=model.hidden_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="sigmoid_direct_quantity",
        activation="selu",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_nn_gated_sigmoid_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(42)
    model = PolicyNet(
        input_dim=4,
        hidden_dim=[8],
        output_dim=21,
        activation="selu",
        action_output_mode="gated_sigmoid_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        hidden_dims=model.hidden_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="gated_sigmoid_direct_quantity",
        activation="selu",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_nn_hard_gated_direct_rollout_matches_python_on_fixed_path():
    np.random.seed(43)
    model = PolicyNet(
        input_dim=4,
        hidden_dim=[8],
        output_dim=21,
        activation="selu",
        action_output_mode="hard_gated_direct_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        hidden_dims=model.hidden_dim,
        output_dim=model.policy_output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="hard_gated_direct_quantity",
        activation="selu",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)


def test_rust_nn_gated_direct_rollout_ignores_policy_cap():
    flat_params = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 30.0], dtype=np.float32)

    uncapped_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=1,
        hidden_dims=[1],
        output_dim=2,
        policy_max_quantity=None,
        policy_head="soft_gated_direct_quantity",
        activation="relu",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )
    capped_argument_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=1,
        hidden_dims=[1],
        output_dim=2,
        policy_max_quantity=20,
        policy_head="soft_gated_direct_quantity",
        activation="relu",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert uncapped_cost == pytest.approx(capped_argument_cost)
    assert uncapped_cost == pytest.approx(15.0)


def test_rust_nn_gated_ordinal_rollout_matches_python_on_fixed_path():
    np.random.seed(41)
    model = PolicyNet(
        input_dim=4,
        hidden_dim=[8],
        output_dim=21,
        activation="selu",
        action_output_mode="soft_gated_ordinal_quantity",
        max_order_size=20,
    )
    flat_params = model.get_model_flat_params().astype(np.float32)

    current_inventory = 7
    lead_time_orders = [2, 4, 1, 3]
    demands = [5, 2, 7, 4, 3, 9, 2, 1, 6, 5]

    python_cost = _run_python_rollout_from_demands(model, current_inventory, lead_time_orders, demands)

    rust_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params.tolist(),
        input_dim=model.input_dim,
        hidden_dims=model.hidden_dim,
        output_dim=model.output_dim,
        policy_max_quantity=model.max_order_size,
        policy_head="soft_gated_ordinal_quantity",
        activation="selu",
        current_inventory=current_inventory,
        lead_time_orders=lead_time_orders,
        demands=demands,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert rust_cost == pytest.approx(python_cost)
