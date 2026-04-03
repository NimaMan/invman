from types import SimpleNamespace

import pytest
import torch

from invman.policies import (
    apply_policy_name,
    make_dense_policy_name,
    make_soft_tree_policy_name,
)
from invman.problems.lost_sales.env import LostSalesEnv
from invman.problems.dual_sourcing.env import DualSourcingEnv
from invman.problems.lost_sales_fixed_order_cost.env import build_env_from_args as build_fixed_cost_env_from_args
from invman.policies import LinearPolicyNet, PolicyNet, SoftTreePolicy, build_policy


def _make_args(policy_name):
    args = SimpleNamespace(
        policy_name=policy_name,
        problem="lost_sales",
        state_features="pipeline",
    )
    apply_policy_name(args)
    return args


def test_build_policy_returns_linear_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear_categorical_quantity"), env)
    assert isinstance(model, LinearPolicyNet)


def test_build_policy_returns_neural_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn_soft_gated_ordinal_quantity"), env)
    assert isinstance(model, PolicyNet)


def test_build_policy_returns_linear_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear_direct_quantity"), env)
    assert isinstance(model, LinearPolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_nn_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn_direct_quantity_h50_selu"), env)
    assert isinstance(model, PolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


@pytest.mark.parametrize(
    "policy_name",
    ["linear_unbounded_direct_quantity", "nn_unbounded_direct_quantity"],
)
def test_unbounded_direct_policy_names_are_not_supported(policy_name):
    with pytest.raises(ValueError, match="Unknown"):
        _make_args(policy_name)


def test_build_policy_returns_linear_gated_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear_soft_gated_direct_quantity"), env)
    assert isinstance(model, LinearPolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_nn_gated_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn_soft_gated_direct_quantity"), env)
    assert isinstance(model, PolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_linear_gated_sigmoid_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear_gated_sigmoid_direct_quantity"), env)
    assert isinstance(model, LinearPolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_nn_gated_sigmoid_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn_gated_sigmoid_direct_quantity"), env)
    assert isinstance(model, PolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_linear_hard_gated_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear_hard_gated_direct_quantity"), env)
    assert isinstance(model, LinearPolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_nn_hard_gated_direct_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn_hard_gated_direct_quantity"), env)
    assert isinstance(model, PolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, int)
    assert 0 <= action <= env.max_order_size


def test_build_policy_returns_linear_policy_for_fixed_cost_lost_sales():
    args = _make_args("linear_categorical_quantity")
    args.problem = "lost_sales_fixed_order_cost"
    env_args = SimpleNamespace(
        demand_rate=5.0,
        lead_time=4,
        max_order_size=20,
        one_hot_inventory_upper_bound=200,
        holding_cost=1.0,
        shortage_cost=4.0,
        horizon=10,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        demand_dist_name="Poisson",
        warm_up_periods_ratio=0.2,
        state_features="pipeline",
    )
    env = build_fixed_cost_env_from_args(env_args, track_demand=True)
    model = build_policy(args, env)
    assert isinstance(model, LinearPolicyNet)


def test_build_policy_returns_neural_policy_for_fixed_cost_lost_sales():
    args = _make_args("nn_soft_gated_ordinal_quantity")
    args.problem = "lost_sales_fixed_order_cost"
    env_args = SimpleNamespace(
        demand_rate=5.0,
        lead_time=4,
        max_order_size=20,
        one_hot_inventory_upper_bound=200,
        holding_cost=1.0,
        shortage_cost=4.0,
        horizon=10,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        demand_dist_name="Poisson",
        warm_up_periods_ratio=0.2,
        state_features="pipeline",
    )
    env = build_fixed_cost_env_from_args(env_args, track_demand=True)
    model = build_policy(args, env)
    assert isinstance(model, PolicyNet)


def test_build_policy_returns_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args(make_soft_tree_policy_name(depth=3, temperature=0.3, split_type="oblique", leaf_type="constant"))
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.depth == 3
    assert model.temperature == 0.3
    assert model.split_type == "oblique"
    assert model.leaf_type == "constant"


def test_build_policy_returns_axis_aligned_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args(make_soft_tree_policy_name(depth=3, temperature=0.3, split_type="axis_aligned", leaf_type="constant"))
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.split_type == "axis_aligned"


def test_build_policy_returns_linear_leaf_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args(make_soft_tree_policy_name(depth=3, temperature=0.3, split_type="oblique", leaf_type="linear"))
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.leaf_type == "linear"


def test_build_policy_returns_sigmoid_linear_leaf_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args(
        make_soft_tree_policy_name(depth=3, temperature=0.3, split_type="oblique", leaf_type="sigmoid_linear")
    )
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.leaf_type == "sigmoid_linear"


def test_legacy_policy_aliases_resolve_to_canonical_names():
    args = _make_args("linear_scaled_gated_direct_quantity")
    assert args.policy_name == "linear_gated_sigmoid_direct_quantity"
    assert args.policy_decoder == "gated_sigmoid_direct_quantity"

    tree_args = _make_args("soft_tree_depth1_scaled_linear_leaf")
    assert tree_args.policy_name == "soft_tree_depth1_sigmoid_linear_leaf"
    assert tree_args.tree_leaf_type == "sigmoid_linear"


def test_policy_name_builders_emit_canonical_descriptive_names():
    assert make_dense_policy_name("linear", "scaled_direct_quantity") == "linear_sigmoid_direct_quantity"
    assert (
        make_dense_policy_name("linear", "two_stage_direct_quantity")
        == "linear_hard_gated_direct_quantity"
    )
    assert (
        make_soft_tree_policy_name(depth=2, temperature=0.25, split_type="oblique", leaf_type="scaled_linear")
        == "soft_tree_d2_t0p25_oblique_sigmoid_linear_leaf"
    )


def test_build_policy_supports_bounded_linear_on_vector_action_problem():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args(make_dense_policy_name("linear", "bounded_quantity"))
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    assert isinstance(model, LinearPolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert 0 <= action[0] <= env.regular_max_order_size
    assert 0 <= action[1] <= env.expedited_max_order_size


def test_build_policy_supports_bounded_nn_on_vector_action_problem():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args(make_dense_policy_name("nn", "bounded_quantity", hidden_dim=[8, 8], activation="relu"))
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    assert isinstance(model, PolicyNet)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert 0 <= action[0] <= env.regular_max_order_size
    assert 0 <= action[1] <= env.expedited_max_order_size


def test_build_policy_supports_soft_tree_on_vector_action_problem():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args(make_soft_tree_policy_name(depth=3, temperature=0.3, split_type="oblique", leaf_type="constant"))
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.action_dim == 2


def test_build_policy_supports_structured_dual_sourcing_tree():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args(
        make_soft_tree_policy_name(
            depth=3,
            temperature=0.3,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="capped_dual_index_targets",
        )
    )
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.action_adapter == "dual_sourcing_capped_dual_index_targets"
    assert model.action_dim == 2
    assert model.control_dim == 3


def test_build_policy_supports_structured_dual_sourcing_nn():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args(
        make_dense_policy_name(
            "nn",
            "bounded_quantity",
            hidden_dim=[8, 8],
            activation="relu",
            action_adapter="base_surge_targets",
        )
    )
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    action = model(torch.as_tensor(env.policy_state, dtype=torch.float32))
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert 0 <= action[0] <= env.regular_max_order_size
    assert 0 <= action[1] <= env.expedited_max_order_size
