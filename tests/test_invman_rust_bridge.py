from types import SimpleNamespace

import numpy as np
import pytest

import invman_rust
from invman.policy import Policy
from invman.rollout_fitness import get_model_fitness, get_population_fitness


def _lost_sales_args(*, demand_dist_name="Poisson"):
    args = SimpleNamespace(
        problem="lost_sales",
        demand_rate=5.0,
        demand_dist_name=demand_dist_name,
        lead_time=4,
        max_order_size=20,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        horizon=80,
        warm_up_periods_ratio=0.2,
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
    )
    return args


def _policy_common(max_order_size=20):
    return dict(
        input_dim=4,
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(20,),
        max_order_size=max_order_size,
        action_adapter="identity",
        state_normalizer="quantity_scale",
        state_scale=20.0,
    )


def _soft_tree_policy(*, leaf_type="linear", max_order_size=20):
    return Policy(
        backbone="soft_tree",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type=leaf_type,
        **_policy_common(max_order_size=max_order_size),
    )


def _linear_policy(*, policy_head="categorical_quantity", output_dim=21, max_order_size=20):
    return Policy(
        backbone="linear",
        output_dim=output_dim,
        action_output_mode=policy_head,
        **_policy_common(max_order_size=max_order_size),
    )


def _nn_policy(*, policy_head="categorical_quantity", output_dim=21, max_order_size=20):
    return Policy(
        backbone="nn",
        output_dim=output_dim,
        action_output_mode=policy_head,
        hidden_dim=(5,),
        activation_name="selu",
        **_policy_common(max_order_size=max_order_size),
    )


def _params_batch(policy):
    base = policy.get_model_flat_params()
    shifted = np.linspace(-0.25, 0.25, policy.num_params, dtype=np.float32)
    return [base, shifted]


def _assert_population_matches_single(policy, args):
    seeds = [11, 22]
    batch = _params_batch(policy)

    population = get_population_fitness(policy, args, batch, seeds)
    singles = [
        get_model_fitness(policy, args, model_params=params, seed=seed, indiv_idx=idx)
        for idx, (params, seed) in enumerate(zip(batch, seeds))
    ]

    assert [idx for _, idx in population] == [0, 1]
    assert [idx for _, idx in singles] == [0, 1]
    assert [score for score, _ in population] == pytest.approx(
        [score for score, _ in singles]
    )


def test_soft_tree_population_fitness_matches_single_model_fitness():
    _assert_population_matches_single(_soft_tree_policy(), _lost_sales_args())


def test_linear_population_fitness_matches_single_model_fitness():
    _assert_population_matches_single(_linear_policy(), _lost_sales_args())


def test_nn_population_fitness_matches_single_model_fitness():
    _assert_population_matches_single(_nn_policy(), _lost_sales_args())


@pytest.mark.parametrize("demand_dist_name", ["Geometric", "MarkovModulatedPoisson2"])
def test_population_fitness_routes_non_poisson_lost_sales_demands_to_rust(demand_dist_name):
    policy = _linear_policy()
    fitness = get_population_fitness(
        policy,
        _lost_sales_args(demand_dist_name=demand_dist_name),
        _params_batch(policy),
        seeds=[101, 202],
    )

    assert len(fitness) == 2
    assert all(np.isfinite(score) and idx in {0, 1} for score, idx in fitness)


def test_rollout_fitness_enforces_policy_state_scale_before_calling_rust():
    policy = _linear_policy()
    policy.state_normalizer = "quantity_scale"
    policy.state_scale = None

    with pytest.raises(ValueError, match="requires an explicit state_scale"):
        get_model_fitness(policy, _lost_sales_args(), seed=11)


def test_rollout_fitness_requires_caps_for_bounded_dense_and_sigmoid_tree_heads():
    dense = _linear_policy(policy_head="sigmoid_direct_quantity", output_dim=1, max_order_size=None)
    tree = _soft_tree_policy(leaf_type="sigmoid_linear", max_order_size=None)

    with pytest.raises(ValueError, match="requires a policy-side quantity cap"):
        get_model_fitness(dense, _lost_sales_args(), seed=11)

    with pytest.raises(ValueError, match="sigmoid_linear soft-tree leaves require"):
        get_model_fitness(tree, _lost_sales_args(), seed=11)


def _linear_bias_params(input_dim, output_dim, bias):
    return [0.0] * (input_dim * output_dim) + [float(value) for value in bias]


@pytest.mark.parametrize(
    ("policy_head", "output_dim", "policy_max_quantity", "bias", "expected_action"),
    [
        ("categorical_quantity", 21, 20, [0.0] * 7 + [2.0] + [0.0] * 13, 7),
        ("direct_quantity", 1, None, [30.0], 30),
        ("sigmoid_direct_quantity", 1, 20, [10.0], 20),
        ("soft_gated_direct_quantity", 2, None, [10.0, 30.0], 30),
        ("gated_sigmoid_direct_quantity", 2, 20, [10.0, 10.0], 20),
        ("hard_gated_direct_quantity", 2, 20, [10.0, 30.0], 20),
        ("soft_gated_ordinal_quantity", 21, 20, [10.0] + [10.0] * 20, 20),
        ("hard_gated_ordinal_quantity", 21, 20, [10.0] + [10.0] * 20, 20),
    ],
)
def test_linear_from_demands_dense_head_semantics(
    policy_head,
    output_dim,
    policy_max_quantity,
    bias,
    expected_action,
):
    cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=_linear_bias_params(1, output_dim, bias),
        input_dim=1,
        output_dim=output_dim,
        policy_max_quantity=policy_max_quantity,
        policy_head=policy_head,
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=0.0,
        warm_up_periods_ratio=0.0,
    )

    assert cost == pytest.approx(expected_action / 2.0)


def test_soft_gated_direct_linear_head_does_not_cap_the_quantity_in_rust():
    flat_params = _linear_bias_params(1, 2, [10.0, 30.0])

    uncapped_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params,
        input_dim=1,
        output_dim=2,
        policy_max_quantity=None,
        policy_head="soft_gated_direct_quantity",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        warm_up_periods_ratio=0.0,
    )
    capped_argument_cost = invman_rust.lost_sales_linear_rollout_from_demands(
        flat_params=flat_params,
        input_dim=1,
        output_dim=2,
        policy_max_quantity=20,
        policy_head="soft_gated_direct_quantity",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        warm_up_periods_ratio=0.0,
    )

    assert uncapped_cost == pytest.approx(15.0)
    assert capped_argument_cost == pytest.approx(uncapped_cost)


def test_soft_gated_direct_nn_head_does_not_cap_the_quantity_in_rust():
    flat_params = [0.0, 0.0, 0.0, 0.0, 10.0, 30.0]

    uncapped_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params,
        input_dim=1,
        hidden_dims=[1],
        output_dim=2,
        policy_max_quantity=None,
        policy_head="soft_gated_direct_quantity",
        activation="relu",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        warm_up_periods_ratio=0.0,
    )
    capped_argument_cost = invman_rust.lost_sales_nn_rollout_from_demands(
        flat_params=flat_params,
        input_dim=1,
        hidden_dims=[1],
        output_dim=2,
        policy_max_quantity=20,
        policy_head="soft_gated_direct_quantity",
        activation="relu",
        current_inventory=0,
        lead_time_orders=[0],
        demands=[0, 0],
        warm_up_periods_ratio=0.0,
    )

    assert uncapped_cost == pytest.approx(15.0)
    assert capped_argument_cost == pytest.approx(uncapped_cost)
