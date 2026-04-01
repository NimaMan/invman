from pathlib import Path

from invman.policies.registry import apply_policy_name
from invman.problems.lost_sales.reference_instances import build_reference_args


COMMON_BUDGET = {
    "training_episodes_default": 2000,
    "training_episodes_lead_time_2": 5000,
    "es_population": 50,
    "horizon_default": 2000,
    "horizon_lead_time_2": 5000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
    "save_every": 1000,
}


EXPERIMENT_SPECS = [
    {
        "id": "linear_categorical_quantity_q8",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "linear_categorical_quantity_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "nn_categorical_quantity_q8",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "nn_categorical_quantity_q20",
        "rollout_backend": "rust",
        "status": "trusted",
    },
    {
        "id": "soft_tree_depth2_linear_leaf_q8",
        "rollout_backend": "rust",
        "status": "trusted",
    },
]


def _resolve_budget(args):
    if int(args.lead_time) == 2:
        return {
            "training_episodes": COMMON_BUDGET["training_episodes_lead_time_2"],
            "horizon": COMMON_BUDGET["horizon_lead_time_2"],
        }
    return {
        "training_episodes": COMMON_BUDGET["training_episodes_default"],
        "horizon": COMMON_BUDGET["horizon_default"],
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
    budget = _resolve_budget(args)
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
    args.policy_name = spec["id"]
    apply_policy_name(args)
    args.rollout_backend = spec["rollout_backend"]
    if args.demand_dist_name != "Poisson":
        args.rollout_backend = "python"
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
