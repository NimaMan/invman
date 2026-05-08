from pathlib import Path

from invman.policies.registry import apply_policy_name
from invman.problems.lost_sales.reference_instances import build_reference_args


COMMON_BUDGET = {
    "training_episodes_default": 2000,
    "es_population": 64,
    "horizon_default": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
    "save_every": 1000,
}


EXPERIMENT_SPECS = [
    {
        "id": "linear_categorical_quantity_q20",
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
        "id": "nn_categorical_quantity_q20",
        "rollout_backend": "rust",
        "status": "provisional",
    },
    {
        "id": "nn_soft_gated_ordinal_quantity",
        "rollout_backend": "rust",
        "status": "provisional",
    },
    {
        "id": "soft_tree_depth1_linear_leaf",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth2_linear_leaf",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth1_sigmoid_linear_leaf_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth2_sigmoid_linear_leaf_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
]


def _resolve_budget(parsed, args):
    del args
    if getattr(parsed, "training_episodes", None) is not None:
        training_episodes = int(parsed.training_episodes)
    else:
        training_episodes = COMMON_BUDGET["training_episodes_default"]

    if getattr(parsed, "training_horizon", None) is not None:
        horizon = int(parsed.training_horizon)
    else:
        horizon = COMMON_BUDGET["horizon_default"]

    return {
        "training_episodes": training_episodes,
        "horizon": horizon,
    }


def resolved_protocol_budget(parsed) -> dict:
    if getattr(parsed, "training_episodes", None) is not None:
        training_episodes_default = int(parsed.training_episodes)
    else:
        training_episodes_default = COMMON_BUDGET["training_episodes_default"]

    if getattr(parsed, "training_horizon", None) is not None:
        horizon_default = int(parsed.training_horizon)
    else:
        horizon_default = COMMON_BUDGET["horizon_default"]

    return {
        "training_episodes_default": training_episodes_default,
        "horizon_default": horizon_default,
    }


def configure_run_args(
    parsed,
    spec,
    root: Path,
    reference_name: str,
    *,
    include_reference_in_experiment_name: bool = True,
):
    args = build_reference_args(reference_name)
    budget = _resolve_budget(parsed, args)
    args.problem = "lost_sales"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes"]
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = budget["horizon"]
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
    args.save_every = COMMON_BUDGET["save_every"]
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
