from pathlib import Path

from invman.policies.registry import apply_policy_name, make_soft_tree_policy_name
from invman.problems.dual_sourcing.reference_instances import build_reference_args


COMMON_BUDGET = {
    "screening": {
        "training_episodes": 300,
        "es_population": 8,
        "es_population_sampling": "fixed",
        "horizon": 1000,
        "eval_horizon": 5000,
        "eval_seeds": 2,
        "sigma_init": 3.0,
    },
    "full": {
        "training_episodes": 1500,
        "es_population": 128,
        "es_population_sampling": "categorical",
        "es_population_candidates": [32, 64, 96, 128],
        "es_population_probabilities": [0.05, 0.15, 0.25, 0.55],
        "horizon": 2000,
        "eval_horizon": 10000,
        "eval_seeds": 3,
        "sigma_init": 3.0,
    },
}

DEFAULT_BUDGET = "screening"


EXPERIMENT_SPECS = [
    {
        "id": "soft_tree_single_index_targets",
        "label": "Soft tree, single-index targets",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="single_index_targets",
        ),
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_dual_index_targets",
        "label": "Soft tree, dual-index targets",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="dual_index_targets",
        ),
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_capped_dual_index_targets",
        "label": "Soft tree, capped dual-index targets",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="capped_dual_index_targets",
        ),
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_base_surge_targets",
        "label": "Soft tree, base-surge targets",
        "policy_name": "soft_tree_base_surge_targets",
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_capped_dual_index_delta_smallcap_targets",
        "label": "Soft tree, small-cap capped dual-index",
        "policy_name": "soft_tree_capped_dual_index_delta_smallcap_targets",
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
        "label": "Soft tree, axis-constant small-cap capped dual-index",
        "policy_name": "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
        "rollout_backend": "rust",
        "status": "selected",
    },
]


def get_budget_config(budget_name: str):
    try:
        return COMMON_BUDGET[budget_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(COMMON_BUDGET))
        raise KeyError(f"Unknown dual-sourcing experiment budget '{budget_name}'. Available: {known}") from exc


def configure_run_args(
    parsed,
    spec,
    root: Path,
    reference_name: str,
    *,
    include_reference_in_experiment_name: bool = True,
):
    budget = get_budget_config(getattr(parsed, "budget", DEFAULT_BUDGET))
    args = build_reference_args(reference_name)
    args.problem = "dual_sourcing"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.es_population_sampling = budget.get("es_population_sampling", "fixed")
    args.es_population_candidates = budget.get("es_population_candidates")
    args.es_population_probabilities = budget.get("es_population_probabilities")
    args.horizon = budget["horizon"]
    args.eval_horizon = int(
        getattr(parsed, "eval_horizon", None)
        if getattr(parsed, "eval_horizon", None) is not None
        else budget["eval_horizon"]
    )
    args.eval_seeds = int(
        getattr(parsed, "eval_seeds", None)
        if getattr(parsed, "eval_seeds", None) is not None
        else budget["eval_seeds"]
    )
    args.sigma_init = float(budget["sigma_init"])
    args.policy_name = spec.get("policy_name", spec["id"])
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
