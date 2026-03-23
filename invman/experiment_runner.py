import json
from copy import copy
from pathlib import Path

import numpy as np

from invman.es_mp import train
from invman.policies import build_policy
from invman.problems import get_problem_module
from invman.utils import set_global_seeds


def build_model(args):
    problem_module = get_problem_module(args.problem)
    env = problem_module.build_env_from_args(args, track_demand=False)
    return build_policy(args, env)


def summarize_costs(costs):
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(len(costs)),
    }


def evaluate_model(model, args):
    problem_module = get_problem_module(args.problem)
    eval_args = copy(args)
    eval_args.horizon = args.eval_horizon
    costs = []
    for seed_offset in range(args.eval_seeds):
        seed = args.seed + seed_offset
        reward, _ = problem_module.get_model_fitness(
            model,
            eval_args,
            seed=seed,
            track_demand=getattr(args, "track_demand", False),
        )
        costs.append(-float(reward))
    return summarize_costs(costs)


def evaluate_heuristics(args):
    problem_module = get_problem_module(args.problem)
    return problem_module.evaluate_default_heuristics(args)


def ensure_output_dirs(args):
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trained_models_dir).mkdir(parents=True, exist_ok=True)


def build_result_payload(args, learned_policy_results, heuristic_results):
    action_adapter = getattr(args, "action_adapter", getattr(args, "tree_action_adapter", "identity"))
    effective_policy_head = (
        args.policy_head
        if args.policy_type != "soft_tree"
        else f"tree_{args.tree_leaf_type}_leaf_quantity"
    )
    adapter_suffix = ""
    if action_adapter != "identity":
        adapter_suffix = f"_{action_adapter}"
    if args.policy_type == "soft_tree":
        policy_architecture = f"{args.policy_type}_{args.tree_split_type}_{effective_policy_head}{adapter_suffix}_{args.state_features}"
    else:
        policy_architecture = f"{args.policy_type}_{effective_policy_head}{adapter_suffix}_{args.state_features}"
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

    return {
        "experiment_name": args.experiment_name,
        "problem": args.problem,
        "problem_params": problem_params,
        "policy_type": args.policy_type,
        "policy_backbone": args.policy_type,
        "policy_head": effective_policy_head,
        "policy_architecture": policy_architecture,
        "action_output_mode": effective_policy_head,
        "state_features": args.state_features,
        "tree_depth": args.tree_depth,
        "tree_temperature": args.tree_temperature,
        "tree_split_type": args.tree_split_type,
        "tree_leaf_type": args.tree_leaf_type,
        "action_adapter": action_adapter,
        "tree_action_adapter": action_adapter,
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
        "training_episodes": args.training_episodes,
        "training_horizon": args.horizon,
        "evaluation_horizon": args.eval_horizon,
        "evaluation": {"learned_policy": learned_policy_results, "heuristics": heuristic_results},
    }


def save_result_payload(args, payload):
    results_path = Path(args.results_dir) / f"{args.experiment_name}.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results_path


def run_experiment(args):
    ensure_output_dirs(args)
    set_global_seeds(getattr(args, "seed", 0))
    problem_module = get_problem_module(args.problem)
    model = build_model(args)
    trained_model, _ = train(
        model=model,
        get_model_fitness=problem_module.get_model_fitness,
        get_population_fitness=problem_module.get_population_fitness,
        args=args,
        same_seed=args.same_seed,
        limit_env_time=args.dynamic_horizon,
        min_steps=args.min_dynamic_horizon,
        max_steps=args.max_dynamic_horizon,
    )

    learned_policy_results = evaluate_model(trained_model, args)
    heuristic_results = evaluate_heuristics(args)
    payload = build_result_payload(args, learned_policy_results, heuristic_results)
    results_path = save_result_payload(args, payload)
    return payload, results_path
