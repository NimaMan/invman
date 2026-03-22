import numpy as np
import torch

from invman.tree_policy import SoftTreePolicy


def test_soft_tree_policy_returns_valid_action_and_features():
    model = SoftTreePolicy(input_dim=4, max_order_size=20, depth=2, temperature=0.25)
    state = torch.tensor(np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32))

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
    state = torch.tensor(np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32))

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
    state = torch.tensor(np.array([0.5, 0.1, 0.2, 0.0], dtype=np.float32))

    action, features = model(state, return_features=True)

    assert 0 <= action <= 20
    assert features["leaf_type"] == "linear"
    assert features["raw_leaf_output"].shape == (4,)
    assert features["leaf_quantities"].shape == (4,)
