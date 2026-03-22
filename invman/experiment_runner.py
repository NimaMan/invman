import json
from copy import copy
from pathlib import Path

import numpy as np

from invman.es_mp import train
from invman.policies import build_policy
from invman.problems import get_problem_module


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
        _, env = problem_module.get_model_fitness(
            model,
            eval_args,
            seed=seed,
            return_env=True,
            track_demand=getattr(args, "track_demand", False),
        )
        costs.append(env.avg_total_cost)
    return summarize_costs(costs)


def evaluate_heuristics(args):
    problem_module = get_problem_module(args.problem)
    return problem_module.evaluate_default_heuristics(args)


def ensure_output_dirs(args):
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trained_models_dir).mkdir(parents=True, exist_ok=True)


def build_result_payload(args, learned_policy_results, heuristic_results):
    effective_policy_head = (
        args.policy_head
        if args.policy_type != "soft_tree"
        else f"tree_{args.tree_leaf_type}_leaf_quantity"
    )
    if args.policy_type == "soft_tree":
        policy_architecture = f"{args.policy_type}_{args.tree_split_type}_{effective_policy_head}_{args.state_features}"
    else:
        policy_architecture = f"{args.policy_type}_{effective_policy_head}_{args.state_features}"
    return {
        "experiment_name": args.experiment_name,
        "problem": args.problem,
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
        "rollout_backend": args.rollout_backend,
        "demand_dist_name": args.demand_dist_name,
        "demand_rate": args.demand_rate,
        "lead_time": args.lead_time,
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
