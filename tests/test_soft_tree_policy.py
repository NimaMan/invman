import math

import numpy as np
import pytest

import invman_rust

from invman.config import get_config
from invman.policy import Policy
from invman.policy_build import build_policy
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name


def _lost_sales_soft_tree_policy(*, leaf_type="linear", split_type="oblique"):
    return Policy(
        backbone="soft_tree",
        input_dim=4,
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(20,),
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type=split_type,
        leaf_type=leaf_type,
        state_normalizer="quantity_scale",
        state_scale=20.0,
    )


def test_soft_tree_policy_descriptor_has_expected_linear_leaf_parameter_count():
    policy = _lost_sales_soft_tree_policy(leaf_type="linear")

    assert policy.num_params == 35
    assert policy.get_model_flat_params().shape == (35,)
    assert policy.to_artifact()["leaf_type"] == "linear"


def test_linear_leaf_soft_tree_action_uses_rust_flat_param_binding():
    policy = _lost_sales_soft_tree_policy(leaf_type="linear")
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action = invman_rust.soft_tree_action_from_flat_params(
        state=state.tolist(),
        flat_params=policy.get_model_flat_params().tolist(),
        input_dim=policy.input_dim,
        depth=policy.depth,
        temperature=policy.temperature,
        split_type=policy.split_type,
        leaf_type=policy.leaf_type,
    )

    assert action == 1


def test_soft_tree_action_vector_binding_matches_current_policy_descriptor():
    policy = _lost_sales_soft_tree_policy(leaf_type="linear")
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action = invman_rust.soft_tree_action_vector_from_flat_params(
        state=state.tolist(),
        flat_params=policy.get_model_flat_params().tolist(),
        input_dim=policy.input_dim,
        depth=policy.depth,
        temperature=policy.temperature,
        split_type=policy.split_type,
        leaf_type=policy.leaf_type,
        control_mode=policy.control_mode,
        min_values=list(policy.min_values),
        max_values=list(policy.max_values),
        allowed_values=policy.allowed_values,
    )

    assert action == [1]


def test_linear_policy_action_binding_uses_current_dense_descriptor():
    input_dim = 2
    output_dim = 3
    weights = np.zeros(input_dim * output_dim, dtype=np.float32)
    bias = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)

    action = invman_rust.linear_policy_action_from_flat_params(
        state=[0.0, 0.0],
        flat_params=np.concatenate([weights, bias]).tolist(),
        input_dim=input_dim,
        output_dim=output_dim,
        policy_head="categorical_quantity",
        policy_max_quantity=None,
    )

    assert action == 1


def test_nn_policy_action_binding_uses_current_dense_descriptor():
    input_dim = 2
    hidden_dims = [2]
    output_dim = 1
    hidden_weights = np.zeros(input_dim * hidden_dims[0], dtype=np.float32)
    hidden_bias = np.zeros(hidden_dims[0], dtype=np.float32)
    output_weights = np.zeros(hidden_dims[0] * output_dim, dtype=np.float32)
    output_bias = np.asarray([math.log(math.exp(3.0) - 1.0)], dtype=np.float32)

    action = invman_rust.nn_policy_action_from_flat_params(
        state=[0.0, 0.0],
        flat_params=np.concatenate(
            [hidden_weights, hidden_bias, output_weights, output_bias]
        ).tolist(),
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation="relu",
        policy_head="direct_quantity",
        policy_max_quantity=None,
    )

    assert action == 3


def test_axis_aligned_capped_soft_tree_action_binding_returns_valid_action():
    state = [0.5, 0.1, 0.2, 0.0]
    split_weights = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    split_bias = [0.0, 0.0, 0.0]
    leaf_logits = [-2.0, -0.5, 0.5, 2.0]

    action = invman_rust.soft_tree_action(
        state=state,
        split_weights=split_weights,
        split_bias=split_bias,
        leaf_logits=leaf_logits,
        depth=2,
        policy_max_quantity=20,
        temperature=0.25,
        split_type="axis_aligned",
    )

    assert 0 <= action <= 20


def test_policy_artifact_round_trips_soft_tree_descriptor(tmp_path):
    policy = _lost_sales_soft_tree_policy(leaf_type="linear")
    policy.flat_params = np.arange(policy.num_params, dtype=np.float32)
    model_dir = tmp_path / "policy"

    policy.save(model_dir)
    loaded = Policy.load(model_dir)

    assert loaded.backbone == "soft_tree"
    assert loaded.input_dim == policy.input_dim
    assert loaded.depth == policy.depth
    assert loaded.leaf_type == policy.leaf_type
    np.testing.assert_allclose(loaded.get_model_flat_params(), policy.get_model_flat_params())


@pytest.mark.parametrize(
    ("adapter", "expected_control_dim", "expected_mode"),
    [
        ("capped_dual_index_targets", 3, "vector_quantity"),
        ("capped_dual_index_delta_targets", 3, "vector_quantity"),
        ("dual_index_delta_targets", 2, "discrete_grid"),
        ("capped_dual_index_delta_smallcap_targets", 3, "discrete_grid"),
    ],
)
def test_structured_dual_sourcing_soft_tree_policy_specs_build_current_descriptor(
    adapter,
    expected_control_dim,
    expected_mode,
):
    args = get_config([])
    args.problem = "dual_sourcing"
    args.policy_name = make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        action_adapter=adapter,
    )
    apply_policy_name(args)

    policy = build_policy(args)

    assert policy.backbone == "soft_tree"
    assert policy.control_dim == expected_control_dim
    assert policy.control_mode == expected_mode
    assert policy.action_adapter.startswith("dual_sourcing_")
    assert policy.num_params > 0
