import json
from copy import copy
from pathlib import Path

import numpy as np

from invman import rollout_fitness
from invman.es_mp import train
from invman.policy_build import build_policy
from invman.policy_registry import apply_policy_name, get_policy_spec
from invman.utils import RunStatusTracker, experiment_status_path, set_global_seeds


def build_model(args):
    apply_policy_name(args)
    return build_policy(args)


def summarize_costs(costs):
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(len(costs)),
    }


def evaluate_model(model, args):
    eval_args = copy(args)
    eval_args.horizon = args.eval_horizon
    costs = []
    for seed_offset in range(args.eval_seeds):
        seed = args.seed + seed_offset
        reward, _ = rollout_fitness.get_model_fitness(model, eval_args, seed=seed)
        costs.append(-float(reward))
    return summarize_costs(costs)


def ensure_output_dirs(args):
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trained_models_dir).mkdir(parents=True, exist_ok=True)


def build_result_payload(args, learned_policy_results, heuristic_results, training_metadata=None):
    policy_spec = get_policy_spec(args)
    policy_architecture = policy_spec.architecture_label(getattr(args, "state_features", "canonical"))
    problem_params = {
        "lead_time": getattr(args, "lead_time", None),
        "fixed_order_cost": getattr(args, "fixed_order_cost", None),
        "regular_lead_time": getattr(args, "regular_lead_time", None),
        "expedited_lead_time": getattr(args, "expedited_lead_time", None),
        "regular_order_cost": getattr(args, "regular_order_cost", None),
        "expedited_order_cost": getattr(args, "expedited_order_cost", None),
        "warehouse_lead_time": getattr(args, "warehouse_lead_time", None),
        "retailer_lead_time": getattr(args, "retailer_lead_time", None),
        "num_retailers": getattr(args, "num_retailers", None),
        "warehouse_capacity": getattr(args, "warehouse_capacity", None),
        "warehouse_inventory_cap": getattr(args, "warehouse_inventory_cap", None),
        "retailer_inventory_cap": getattr(args, "retailer_inventory_cap", None),
        "multi_demand_mean": getattr(args, "multi_demand_mean", None),
        "multi_demand_std": getattr(args, "multi_demand_std", None),
        "dual_demand_low": getattr(args, "dual_demand_low", None),
        "dual_demand_high": getattr(args, "dual_demand_high", None),
    }
    problem_params = {key: value for key, value in problem_params.items() if value is not None}
    es_population_protocol = None if training_metadata is None else training_metadata.get("es_population_protocol")
    if es_population_protocol is None:
        es_population_protocol = {
            "base_population": int(args.es_population),
            "sampling_mode": str(getattr(args, "es_population_sampling", "fixed")),
            "candidates": getattr(args, "es_population_candidates", None),
            "probabilities": getattr(args, "es_population_probabilities", None),
        }

    return {
        "experiment_name": args.experiment_name,
        "problem": args.problem,
        "problem_params": problem_params,
        "policy_name": policy_spec.policy_name,
        "policy_backbone": policy_spec.policy_backbone,
        "policy_decoder": policy_spec.action_output_mode,
        "policy_architecture": policy_architecture,
        "state_features": getattr(args, "state_features", None),
        "state_normalizer": getattr(args, "state_normalizer", None),
        "state_scale": getattr(args, "state_scale", None),
        "hidden_dim": list(policy_spec.hidden_dim) if policy_spec.hidden_dim else None,
        "activation": policy_spec.activation,
        "tree_depth": policy_spec.tree_depth,
        "tree_temperature": policy_spec.tree_temperature,
        "tree_split_type": policy_spec.tree_split_type,
        "tree_leaf_type": policy_spec.tree_leaf_type,
        "action_adapter": policy_spec.action_adapter,
        "rollout_backend": args.rollout_backend,
        "demand_dist_name": args.demand_dist_name,
        "demand_rate": args.demand_rate,
        "max_order_size": args.max_order_size,
        "holding_cost": args.holding_cost,
        "shortage_cost": args.shortage_cost,
        "procurement_cost": args.procurement_cost,
        "fixed_order_cost": args.fixed_order_cost,
        "training_method": args.training_method,
        "parameter_optimizer": args.training_method,
        "es_population": int(args.es_population),
        "es_population_protocol": es_population_protocol,
        "training_episodes": args.training_episodes,
        "training_horizon": args.horizon,
        "dynamic_horizon": bool(getattr(args, "dynamic_horizon", False)),
        "min_dynamic_horizon": int(getattr(args, "min_dynamic_horizon", args.horizon)),
        "max_dynamic_horizon": int(getattr(args, "max_dynamic_horizon", args.horizon)),
        "evaluation_horizon": args.eval_horizon,
        "evaluation": {"learned_policy": learned_policy_results, "heuristics": heuristic_results},
    }


def save_result_payload(args, payload):
    results_path = Path(args.results_dir) / f"{args.experiment_name}.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results_path


def run_experiment(args):
    apply_policy_name(args)
    ensure_output_dirs(args)
    policy_spec = get_policy_spec(args)
    status_metadata = {
        "experiment_name": args.experiment_name,
        "problem": args.problem,
        "policy_name": policy_spec.policy_name,
        "policy_decoder": policy_spec.action_output_mode,
        "seed": int(getattr(args, "seed", 0)),
    }
    with RunStatusTracker(experiment_status_path(args), metadata=status_metadata) as tracker:
        tracker.update("seeding")
        set_global_seeds(getattr(args, "seed", 0))
        tracker.update("building_model")
        model = build_model(args)
        tracker.update("training")
        trained_model, _ = train(
            model=model,
            get_model_fitness=rollout_fitness.get_model_fitness,
            get_population_fitness=rollout_fitness.get_population_fitness,
            args=args,
            same_seed=args.same_seed,
            limit_env_time=args.dynamic_horizon,
            min_steps=args.min_dynamic_horizon,
            max_steps=args.max_dynamic_horizon,
        )
        training_metadata = getattr(trained_model, "training_run_metadata", None)

        tracker.update("evaluating_learned_policy")
        learned_policy_results = evaluate_model(trained_model, args)
        # Heuristic baselines are computed in Rust now; the Python rollout/heuristics
        # were removed in the Python-cleanup migration.
        heuristic_results = {}
        tracker.update("writing_results")
        payload = build_result_payload(
            args,
            learned_policy_results,
            heuristic_results,
            training_metadata=training_metadata,
        )
        results_path = save_result_payload(args, payload)
        tracker.mark_completed(results_path=str(results_path))
        return payload, results_path
