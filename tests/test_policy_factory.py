from types import SimpleNamespace

import pytest

from invman.problems.lost_sales.env import LostSalesEnv
from invman.problems.dual_sourcing.env import DualSourcingEnv
from invman.policies import LinearPolicyNet, PolicyNet, SoftTreePolicy, build_policy


def _make_args(policy_type, policy_head="categorical_quantity"):
    return SimpleNamespace(
        policy_type=policy_type,
        problem="lost_sales",
        policy_head=policy_head,
        hidden_dim=[8, 8],
        activation="relu",
        tree_depth=3,
        tree_temperature=0.3,
        tree_split_type="oblique",
        tree_leaf_type="constant",
        tree_action_adapter="identity",
    )


def test_build_policy_returns_linear_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("linear"), env)
    assert isinstance(model, LinearPolicyNet)


def test_build_policy_returns_neural_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("nn", policy_head="gated_ordinal_quantity"), env)
    assert isinstance(model, PolicyNet)


def test_build_policy_returns_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    model = build_policy(_make_args("soft_tree"), env)
    assert isinstance(model, SoftTreePolicy)
    assert model.depth == 3
    assert model.temperature == 0.3
    assert model.split_type == "oblique"
    assert model.leaf_type == "constant"


def test_build_policy_returns_axis_aligned_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args("soft_tree")
    args.tree_split_type = "axis_aligned"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.split_type == "axis_aligned"


def test_build_policy_returns_linear_leaf_soft_tree_policy():
    env = LostSalesEnv(demand_rate=5.0, lead_time=4, max_order_size=20, horizon=10, track_demand=True)
    args = _make_args("soft_tree")
    args.tree_leaf_type = "linear"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.leaf_type == "linear"


def test_build_policy_rejects_linear_on_vector_action_problem():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    with pytest.raises(ValueError):
        build_policy(_make_args("linear"), env)


def test_build_policy_supports_soft_tree_on_vector_action_problem():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args("soft_tree")
    args.problem = "dual_sourcing"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.action_dim == 2


def test_build_policy_supports_structured_dual_sourcing_tree():
    env = DualSourcingEnv(horizon=10, track_demand=True)
    args = _make_args("soft_tree")
    args.problem = "dual_sourcing"
    args.tree_leaf_type = "linear"
    args.tree_action_adapter = "capped_dual_index_targets"
    model = build_policy(args, env)
    assert isinstance(model, SoftTreePolicy)
    assert model.action_adapter == "dual_sourcing_capped_dual_index_targets"
    assert model.action_dim == 2
    assert model.control_dim == 3
