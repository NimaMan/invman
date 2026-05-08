from pathlib import Path

from invman.policies.registry import apply_policy_name
from invman.problems.lost_sales_fixed_order_cost.reference_instances import build_reference_args


COMMON_BUDGET = {
    "training_episodes": 2000,
    "es_population": 64,
    "horizon": 2000,
    "dynamic_horizon": False,
    "min_dynamic_horizon": 2000,
    "max_dynamic_horizon": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
}


EXPERIMENT_SPECS = [
    {
        "id": "linear_categorical_quantity",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "linear_sigmoid_direct_quantity",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "linear_soft_gated_direct_quantity",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "nn_soft_gated_direct_quantity_h8_selu",
        "rollout_backend": "rust",
        "status": "provisional",
    },
    {
        "id": "linear_hard_gated_direct_quantity",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "linear_soft_gated_ordinal_quantity",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "nn_soft_gated_ordinal_quantity_h8_selu",
        "rollout_backend": "rust",
        "status": "provisional",
    },
    {
        "id": "nn_categorical_quantity",
        "rollout_backend": "rust",
        "status": "provisional",
        "note": (
            "Current canonical run matches the linear categorical baseline exactly and should be "
            "re-verified before publication claims rely on it."
        ),
    },
    {
        "id": "nn_soft_gated_ordinal_quantity",
        "rollout_backend": "rust",
        "status": "provisional",
    },
    {
        "id": "soft_tree_depth2_linear_leaf",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth1_linear_leaf",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth2_sigmoid_linear_leaf_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth1_sigmoid_linear_leaf_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
]


def configure_run_args(
    parsed,
    spec,
    root: Path,
    reference_name: str,
    *,
    include_reference_in_experiment_name: bool = True,
):
    args = build_reference_args(reference_name)
    args.problem = "lost_sales_fixed_order_cost"
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = int(
        parsed.training_episodes
        if getattr(parsed, "training_episodes", None) is not None
        else COMMON_BUDGET["training_episodes"]
    )
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = int(
        parsed.training_horizon
        if getattr(parsed, "training_horizon", None) is not None
        else COMMON_BUDGET["horizon"]
    )
    args.dynamic_horizon = COMMON_BUDGET["dynamic_horizon"]
    args.min_dynamic_horizon = COMMON_BUDGET["min_dynamic_horizon"]
    args.max_dynamic_horizon = COMMON_BUDGET["max_dynamic_horizon"]
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
    args.max_order_size = 20
    args.policy_name = spec["id"]
    apply_policy_name(args)
    args.rollout_backend = spec["rollout_backend"]
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    if include_reference_in_experiment_name:
        args.experiment_name = f"{parsed.run_tag}_{reference_name}_{spec['id']}"
    else:
        args.experiment_name = f"{parsed.run_tag}_{spec['id']}"

    return args


def result_path_for(args):
    return Path(args.results_dir) / f"{args.experiment_name}.json"


def resolved_protocol_budget(parsed) -> dict:
    return {
        "training_episodes": int(
            parsed.training_episodes
            if getattr(parsed, "training_episodes", None) is not None
            else COMMON_BUDGET["training_episodes"]
        ),
        "horizon": int(
            parsed.training_horizon
            if getattr(parsed, "training_horizon", None) is not None
            else COMMON_BUDGET["horizon"]
        ),
    }
