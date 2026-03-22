from copy import copy

import numpy as np

from invman.problems.lost_sales.heuristics import get_heuristic_policy_cost


def summarize_costs(costs):
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(len(costs)),
    }


def evaluate_default_heuristics(args):
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
