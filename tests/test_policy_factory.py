from types import SimpleNamespace

import pytest

from invman.policy import Policy
from invman.policy_build import build_policy
from invman.policy_registry import (
    apply_policy_name,
    make_dense_policy_name,
    make_soft_tree_policy_name,
)


def _make_args(policy_name, *, problem="lost_sales", **overrides):
    args = SimpleNamespace(
        policy_name=policy_name,
        problem=problem,
        lead_time=4,
        max_order_size=20,
        state_normalizer="quantity_scale",
        state_scale=None,
        regular_lead_time=2,
        regular_max_order_size=12,
        expedited_max_order_size=8,
        dual_demand_low=0,
        dual_demand_high=4,
        warehouse_lead_time=2,
        retailer_lead_time=2,
        num_retailers=3,
        warehouse_inventory_cap=100,
        retailer_inventory_cap=40,
        multi_action_design="direct_level",
        inventory_dynamics_mode="gijs_2022",
        include_period_feature=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    apply_policy_name(args)
    return args


@pytest.mark.parametrize(
    ("policy_name", "backbone", "head", "output_dim", "hidden_dim"),
    [
        ("linear_categorical_quantity", "linear", "categorical_quantity", 21, ()),
        ("linear_direct_quantity", "linear", "direct_quantity", 1, ()),
        ("linear_capped_direct_quantity", "linear", "capped_direct_quantity", 1, ()),
        ("linear_soft_gated_direct_quantity", "linear", "soft_gated_direct_quantity", 2, ()),
        ("linear_gated_sigmoid_direct_quantity", "linear", "gated_sigmoid_direct_quantity", 2, ()),
        ("linear_hard_gated_direct_quantity", "linear", "hard_gated_direct_quantity", 2, ()),
        ("linear_soft_gated_ordinal_quantity", "linear", "soft_gated_ordinal_quantity", 21, ()),
        ("nn_soft_gated_ordinal_quantity", "nn", "soft_gated_ordinal_quantity", 21, (50,)),
        ("nn_direct_quantity_h50_selu", "nn", "direct_quantity", 1, (50,)),
        ("nn_soft_gated_direct_quantity_h8_selu", "nn", "soft_gated_direct_quantity", 2, (8,)),
    ],
)
def test_build_policy_returns_lost_sales_descriptor_for_dense_policies(
    policy_name,
    backbone,
    head,
    output_dim,
    hidden_dim,
):
    policy = build_policy(_make_args(policy_name))

    assert isinstance(policy, Policy)
    assert policy.backbone == backbone
    assert policy.input_dim == 4
    assert policy.control_dim == 1
    assert policy.control_mode == "scalar_quantity"
    assert policy.max_order_size == 20
    assert policy.output_dim == output_dim
    assert policy.action_output_mode == head
    assert policy.hidden_dim == hidden_dim
    assert policy.num_params == policy.get_model_flat_params().size


@pytest.mark.parametrize("problem", ["lost_sales", "lost_sales_fixed_order_cost"])
@pytest.mark.parametrize("policy_name", ["linear_categorical_quantity", "nn_soft_gated_ordinal_quantity"])
def test_build_policy_uses_same_scalar_descriptor_for_lost_sales_variants(problem, policy_name):
    policy = build_policy(_make_args(policy_name, problem=problem))

    assert policy.input_dim == 4
    assert policy.control_mode == "scalar_quantity"
    assert policy.max_values == (20,)
    assert policy.state_normalizer == "divide_by_scale"
    assert policy.state_scale == 20.0


@pytest.mark.parametrize(
    ("policy_name", "depth", "temperature", "split_type", "leaf_type", "max_order_size"),
    [
        (
            make_soft_tree_policy_name(
                depth=3,
                temperature=0.3,
                split_type="oblique",
                leaf_type="constant",
            ),
            3,
            0.3,
            "oblique",
            "constant",
            20,
        ),
        (
            make_soft_tree_policy_name(
                depth=3,
                temperature=0.3,
                split_type="axis_aligned",
                leaf_type="linear",
            ),
            3,
            0.3,
            "axis_aligned",
            "linear",
            20,
        ),
        (
            make_soft_tree_policy_name(
                depth=2,
                temperature=0.25,
                split_type="oblique",
                leaf_type="sigmoid_linear",
                max_order_size=20,
            ),
            2,
            0.25,
            "oblique",
            "sigmoid_linear",
            20,
        ),
    ],
)
def test_build_policy_returns_lost_sales_descriptor_for_soft_trees(
    policy_name,
    depth,
    temperature,
    split_type,
    leaf_type,
    max_order_size,
):
    policy = build_policy(_make_args(policy_name))

    assert policy.backbone == "soft_tree"
    assert policy.depth == depth
    assert policy.temperature == pytest.approx(temperature)
    assert policy.split_type == split_type
    assert policy.leaf_type == leaf_type
    assert policy.max_order_size == max_order_size
    assert policy.num_params == policy.get_model_flat_params().size


@pytest.mark.parametrize(
    "policy_name",
    [
        "linear_unknown_direct_quantity",
        "nn_unknown_direct_quantity",
    ],
)
def test_unknown_direct_policy_names_are_not_supported(policy_name):
    with pytest.raises(ValueError, match="Unknown"):
        _make_args(policy_name)


def test_legacy_policy_aliases_resolve_to_canonical_names():
    args = _make_args("linear_scaled_gated_direct_quantity")
    assert args.policy_name == "linear_gated_sigmoid_direct_quantity"
    assert args.policy_decoder == "gated_sigmoid_direct_quantity"

    tree_args = _make_args("soft_tree_depth1_scaled_linear_leaf")
    assert tree_args.policy_name == "soft_tree_depth1_sigmoid_linear_leaf"
    assert tree_args.tree_leaf_type == "sigmoid_linear"

    bounded_tree_args = _make_args("soft_tree_depth2_sigmoid_linear_leaf_q20")
    assert bounded_tree_args.policy_name == "soft_tree_depth2_sigmoid_linear_leaf_q20"
    assert bounded_tree_args.tree_leaf_type == "sigmoid_linear"
    assert bounded_tree_args.max_order_size == 20


def test_policy_name_builders_emit_canonical_descriptive_names():
    assert make_dense_policy_name("linear", "scaled_direct_quantity") == "linear_sigmoid_direct_quantity"
    assert (
        make_dense_policy_name("linear", "two_stage_direct_quantity")
        == "linear_hard_gated_direct_quantity"
    )
    assert (
        make_dense_policy_name(
            "nn",
            "bounded_quantity",
            hidden_dim=[8, 8],
            activation="relu",
            action_adapter="base_surge_targets",
        )
        == "nn_bounded_quantity_h8x8_relu_adapter-base_surge_targets"
    )
    assert (
        make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="scaled_linear",
        )
        == "soft_tree_d2_t0p25_oblique_sigmoid_linear_leaf"
    )


@pytest.mark.parametrize(
    ("policy_name", "control_dim", "control_mode"),
    [
        ("soft_tree_identity", 2, "vector_quantity"),
        ("soft_tree_dual_index_delta_targets", 2, "discrete_grid"),
        ("soft_tree_capped_dual_index_delta_targets", 3, "vector_quantity"),
        ("soft_tree_capped_dual_index_delta_smallcap_targets", 3, "discrete_grid"),
        (
            make_soft_tree_policy_name(
                depth=3,
                temperature=0.3,
                split_type="oblique",
                leaf_type="linear",
                action_adapter="capped_dual_index_targets",
            ),
            3,
            "vector_quantity",
        ),
    ],
)
def test_build_policy_returns_structured_dual_sourcing_soft_tree_descriptor(
    policy_name,
    control_dim,
    control_mode,
):
    policy = build_policy(_make_args(policy_name, problem="dual_sourcing"))

    assert policy.backbone == "soft_tree"
    assert policy.input_dim == 2
    assert policy.control_dim == control_dim
    assert policy.control_mode == control_mode
    assert policy.action_adapter.startswith("dual_sourcing_") or policy.action_adapter == "identity"
    assert policy.num_params == policy.get_model_flat_params().size


def test_smallcap_dual_sourcing_descriptor_preserves_allowed_cap_grid():
    policy = build_policy(
        _make_args("soft_tree_capped_dual_index_delta_smallcap_targets", problem="dual_sourcing")
    )

    assert policy.control_mode == "discrete_grid"
    assert policy.allowed_values[2] == [1, 2, 3, 4, 6, 8, 12]


def test_axis_constant_smallcap_dual_sourcing_descriptor_sets_tree_shape_and_adapter():
    policy = build_policy(
        _make_args(
            "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
            problem="dual_sourcing",
        )
    )

    assert policy.action_adapter == "dual_sourcing_capped_dual_index_delta_smallcap_targets"
    assert policy.control_dim == 3
    assert policy.control_mode == "discrete_grid"
    assert policy.split_type == "axis_aligned"
    assert policy.leaf_type == "constant"


@pytest.mark.parametrize(
    "policy_name",
    [
        make_dense_policy_name("linear", "bounded_quantity"),
        make_dense_policy_name("nn", "bounded_quantity", hidden_dim=[8, 8], activation="relu"),
    ],
)
def test_dense_dual_sourcing_policies_are_not_supported_by_current_rust_rollout(policy_name):
    with pytest.raises(NotImplementedError, match="dual_sourcing supports only the soft_tree"):
        build_policy(_make_args(policy_name, problem="dual_sourcing"))
