from invman.policy_build import build_policy
from invman.policy_registry import apply_policy_name
from scripts.lost_sales.benchmark_canonical_suite import (
    build_reference_args as build_lost_sales_reference_args,
)
from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import (
    build_reference_args as build_fixed_cost_reference_args,
)


def _build_model(build_args, reference_name, policy_name):
    args = build_args(reference_name)
    args.policy_name = policy_name
    apply_policy_name(args)
    model = build_policy(args)
    return args, model


def test_l4_pipeline_parameter_counts_match_formulas_for_canonical_instances():
    cases = (
        (
            "vanilla_l4_p4_poisson5",
            build_lost_sales_reference_args,
            20,
        ),
        (
            "lit_pois_mu5_l4_p4_k5",
            build_fixed_cost_reference_args,
            50,
        ),
    )

    for reference_name, build_args, expected_q in cases:
        base_args = build_args(reference_name)
        base_model = build_policy(_with_policy_name(base_args, "linear_direct_quantity"))
        assert base_args.state_features == "pipeline"
        assert base_model.input_dim == 4
        assert int(base_args.max_order_size) == expected_q

        d = int(base_model.input_dim)
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
        h8 = 8
        nn_h8_expected_counts = {
            "nn_soft_gated_direct_quantity_h8_selu": h8 * (d + 1) + 2 * (h8 + 1),
            "nn_soft_gated_ordinal_quantity_h8_selu": h8 * (d + 1) + (h8 + 1) * (q + 1),
        }

        for policy_name, expected_num_params in {
            **linear_expected_counts,
            **nn_expected_counts,
            **nn_h8_expected_counts,
            **tree_expected_counts,
        }.items():
            _, model = _build_model(build_args, reference_name, policy_name)
            assert model.input_dim == d
            if policy_name in nn_h8_expected_counts:
                assert tuple(model.hidden_dim) == (h8,)
            elif policy_name.startswith("nn_"):
                assert tuple(model.hidden_dim) == (h,)
            assert model.num_params == expected_num_params, (
                f"{reference_name} / {policy_name}: expected {expected_num_params} "
                f"parameters, got {model.num_params}"
            )


def _with_policy_name(args, policy_name):
    args.policy_name = policy_name
    apply_policy_name(args)
    return args
