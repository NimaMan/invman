from invman.policies import apply_policy_name, build_policy
from invman.problems.lost_sales.env import build_env_from_args as build_lost_sales_env_from_args
from invman.problems.lost_sales.reference_instances import build_reference_args as build_lost_sales_reference_args
from invman.problems.lost_sales_fixed_order_cost.env import (
    build_env_from_args as build_fixed_cost_env_from_args,
)
from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    CANONICAL_REFERENCE_NAME,
    build_reference_args as build_fixed_cost_reference_args,
)


def _build_model(build_args, build_env, reference_name, policy_name):
    args = build_args(reference_name)
    args.policy_name = policy_name
    apply_policy_name(args)
    env = build_env(args, track_demand=True)
    model = build_policy(args, env)
    return args, env, model


def test_l4_pipeline_parameter_counts_match_formulas_for_canonical_instances():
    cases = (
        (
            "vanilla_l4_p4_poisson5",
            build_lost_sales_reference_args,
            build_lost_sales_env_from_args,
            20,
        ),
        (
            CANONICAL_REFERENCE_NAME,
            build_fixed_cost_reference_args,
            build_fixed_cost_env_from_args,
            50,
        ),
    )

    for reference_name, build_args, build_env, expected_q in cases:
        base_args = build_args(reference_name)
        base_env = build_env(base_args, track_demand=True)
        assert base_args.state_features == "pipeline"
        assert base_env.state_space_dim == 4
        assert int(base_args.max_order_size) == expected_q

        d = int(base_env.state_space_dim)
        q = int(base_args.max_order_size)
        linear_expected_counts = {
            "linear_categorical_quantity": (d + 1) * (q + 1),
            "linear_sigmoid_direct_quantity": d + 1,
            "linear_direct_quantity": d + 1,
            "linear_capped_direct_quantity": d + 1,
            "linear_gated_sigmoid_direct_quantity": 2 * (d + 1),
            "linear_soft_gated_direct_quantity": 2 * (d + 1),
            "linear_hard_gated_direct_quantity": 2 * (d + 1),
            "linear_soft_gated_ordinal_quantity": (d + 1) * (q + 1),
        }
        tree_expected_counts = {
            "soft_tree_depth1_linear_leaf": 3 * (d + 1),
            "soft_tree_depth2_linear_leaf": 7 * (d + 1),
        }
        h = 50
        nn_expected_counts = {
            "nn_categorical_quantity": h * (d + 1) + (h + 1) * (q + 1),
            "nn_sigmoid_direct_quantity": h * (d + 1) + (h + 1),
            "nn_direct_quantity": h * (d + 1) + (h + 1),
            "nn_capped_direct_quantity": h * (d + 1) + (h + 1),
            "nn_gated_sigmoid_direct_quantity": h * (d + 1) + 2 * (h + 1),
            "nn_soft_gated_direct_quantity": h * (d + 1) + 2 * (h + 1),
            "nn_hard_gated_direct_quantity": h * (d + 1) + 2 * (h + 1),
            "nn_soft_gated_ordinal_quantity": h * (d + 1) + (h + 1) * (q + 1),
        }

        for policy_name, expected_num_params in {
            **linear_expected_counts,
            **nn_expected_counts,
            **tree_expected_counts,
        }.items():
            _, env, model = _build_model(build_args, build_env, reference_name, policy_name)
            assert env.state_space_dim == d
            if policy_name.startswith("nn_"):
                assert tuple(model.hidden_dim) == (h,)
            assert model.num_params == expected_num_params, (
                f"{reference_name} / {policy_name}: expected {expected_num_params} "
                f"parameters, got {model.num_params}"
            )
