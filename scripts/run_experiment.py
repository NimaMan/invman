import json
import sys
from copy import copy
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.config import get_config
from invman.env.lost_sales import build_env_from_args, get_model_fitness
from invman.es_mp import train
from invman.heuristics.lost_sales_heuristics import get_heuristic_policy_cost
from invman.nn.linear_policy_net import LinearPolicyNet
from invman.nn.policy_net import PolicyNet


def build_model(args):
    env = build_env_from_args(args, track_demand=False)
    if args.policy_type == "linear":
        return LinearPolicyNet(input_dim=env.state_space_dim, output_dim=env.action_space_dim)
    if args.policy_type == "nn":
        return PolicyNet(
            input_dim=env.state_space_dim,
            hidden_dim=args.hidden_dim,
            output_dim=env.action_space_dim,
            activation=args.activation,
        )
    raise NotImplementedError(f"Unknown policy type: {args.policy_type}")


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
        _, env = get_model_fitness(model, eval_args, seed=seed, return_env=True, track_demand=False)
        costs.append(env.avg_total_cost)
    return summarize_costs(costs)


def evaluate_heuristics(args):
    if args.problem != "lost_sales" or args.fixed_order_cost > 0:
        return {
            "status": "skipped",
            "reason": "Classic lost-sales heuristics are not valid benchmarks once a fixed order cost is introduced.",
        }

    eval_args = copy(args)
    eval_args.horizon = args.eval_horizon
    heuristic_results = {}
    for heuristic_name in ("myopic1", "myopic2", "svbs"):
        costs = []
        for seed_offset in range(args.eval_seeds):
            eval_args.seed = args.seed + seed_offset
            env, _, _ = get_heuristic_policy_cost(eval_args, heuristic=heuristic_name)
            costs.append(env.avg_total_cost)
        heuristic_results[heuristic_name] = summarize_costs(costs)
    return heuristic_results


def ensure_output_dirs(args):
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trained_models_dir).mkdir(parents=True, exist_ok=True)


def main():
    args = get_config()
    ensure_output_dirs(args)
    model = build_model(args)
    trained_model, _ = train(
        model=model,
        get_model_fitness=get_model_fitness,
        args=args,
        same_seed=args.same_seed,
        limit_env_time=args.dynamic_horizon,
        min_steps=args.min_dynamic_horizon,
        max_steps=args.max_dynamic_horizon,
    )

    learned_policy_results = evaluate_model(trained_model, args)
    heuristic_results = evaluate_heuristics(args)

    result_payload = {
        "experiment_name": args.experiment_name,
        "problem": args.problem,
        "policy_type": args.policy_type,
        "demand_dist_name": args.demand_dist_name,
        "demand_rate": args.demand_rate,
        "lead_time": args.lead_time,
        "max_order_size": args.max_order_size,
        "holding_cost": args.holding_cost,
        "shortage_cost": args.shortage_cost,
        "procurement_cost": args.procurement_cost,
        "fixed_order_cost": args.fixed_order_cost,
        "training_method": args.training_method,
        "training_episodes": args.training_episodes,
        "training_horizon": args.horizon,
        "evaluation_horizon": args.eval_horizon,
        "evaluation": {"learned_policy": learned_policy_results, "heuristics": heuristic_results},
    }

    results_path = Path(args.results_dir) / f"{args.experiment_name}.json"
    results_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(json.dumps(result_payload["evaluation"], indent=2))
    print(f"saved results to {results_path}")


if __name__ == "__main__":
    main()
