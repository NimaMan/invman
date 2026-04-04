import numpy as np

from invman.policies import SoftTreePolicy


def test_soft_tree_policy_returns_valid_action_and_features():
    model = SoftTreePolicy(input_dim=4, max_order_size=20, depth=2, temperature=0.25)
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action, features = model(state, return_features=True)

    assert 0 <= action <= 20
    assert features["split_probs"].shape == (3,)
    assert features["leaf_probs"].shape == (4,)
    assert np.isclose(features["leaf_probs"].sum(), 1.0)
    assert features["leaf_quantities"].shape == (4,)


def test_axis_aligned_soft_tree_policy_reports_selected_features():
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="axis_aligned",
    )
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action, features = model(state, return_features=True)

    assert 0 <= action <= 20
    assert features["split_type"] == "axis_aligned"
    assert features["selected_feature_idx"].shape == (3,)
    assert features["selected_feature_weight"].shape == (3,)


def test_linear_leaf_soft_tree_policy_reports_leaf_outputs():
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    )
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action, features = model(state, return_features=True)

    assert 0 <= action <= 20
    assert features["leaf_type"] == "linear"
    assert features["raw_leaf_output"].shape == (4,)
    assert features["leaf_quantities"].shape == (4,)
    assert np.all(features["leaf_quantities"] >= 0.0)


def test_sigmoid_linear_leaf_soft_tree_policy_reports_leaf_outputs():
    model = SoftTreePolicy(
        input_dim=4,
        max_order_size=20,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    )
    state = np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32)

    action, features = model(state, return_features=True)

    assert 0 <= action <= 20
    assert features["leaf_type"] == "sigmoid_linear"
    assert features["raw_leaf_output"].shape == (4,)
    assert features["leaf_quantities"].shape == (4,)


def test_structured_dual_sourcing_soft_tree_projects_controls_and_actions():
    model = SoftTreePolicy(
        input_dim=4,
        action_spec={
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [12, 12],
            "allowed_values": None,
        },
        control_spec={
            "action_dim": 3,
            "action_mode": "vector_quantity",
            "min_values": [0, 0, 0],
            "max_values": [24, 24, 12],
            "allowed_values": None,
        },
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        action_adapter="capped_dual_index_targets",
        action_adapter_config={
            "regular_max_order_size": 12,
            "expedited_max_order_size": 12,
            "state_scale": 24.0,
        },
    )
    state = np.array([8.0, 3.0, 1.0, 2.0], dtype=np.float32) / 24.0

    action, features = model(state, return_features=True)

    assert isinstance(action, tuple)
    assert len(action) == 2
    assert 0 <= action[0] <= 12
    assert 0 <= action[1] <= 12
    assert features["action_adapter"] == "dual_sourcing_capped_dual_index_targets"
    assert features["leaf_quantities"].shape == (4, 3)
    assert features["projected_controls"].shape == (3,)
