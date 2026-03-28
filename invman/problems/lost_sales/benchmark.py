from __future__ import annotations

from copy import copy

import numpy as np

from invman.problems.lost_sales.heuristics import get_heuristic_policy_cost
from invman.problems.lost_sales.reference_instances import (
    build_reference_args,
    get_benchmark_grid,
    get_reference_instance,
)


def summarize_costs(costs):
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(len(costs)),
    }


def _evaluate_heuristic(args, heuristic_name: str, *, num_seeds: int, horizon: int, track_demand: bool):
    eval_args = copy(args)
    eval_args.horizon = int(horizon)
    eval_args.track_demand = bool(track_demand)
    costs = []
    max_orders = []
    for seed_offset in range(int(num_seeds)):
        eval_args.seed = int(args.seed + seed_offset)
        env, _, state_action = get_heuristic_policy_cost(eval_args, heuristic=heuristic_name)
        costs.append(env.avg_total_cost)
        max_orders.append(max(state_action.values()) if state_action else 0)
    summary = summarize_costs(costs)
    summary["max_order_observed"] = int(max(max_orders) if max_orders else 0)
    return summary


def evaluate_default_heuristics(args):
    heuristic_args = copy(args)
    reference_instance = getattr(args, "reference_instance", None)
    if reference_instance is not None:
        reference = get_reference_instance(reference_instance)
        heuristic_args.max_order_size = int(reference.heuristic_max_order_size)
    heuristic_args.horizon = args.eval_horizon

    return {
        heuristic_name: _evaluate_heuristic(
            heuristic_args,
            heuristic_name,
            num_seeds=args.eval_seeds,
            horizon=args.eval_horizon,
            track_demand=getattr(args, "track_demand", False),
        )
        for heuristic_name in ("myopic1", "myopic2", "svbs")
    }


def benchmark_reference_instance(
    reference_instance: str,
    *,
    eval_horizon: int | None = None,
    eval_seeds: int | None = None,
    heuristic_max_order_size: int | None = None,
):
    reference = get_reference_instance(reference_instance)
    args = build_reference_args(reference_instance)
    resolved_eval_horizon = reference.params["eval_horizon"] if eval_horizon is None else int(eval_horizon)
    resolved_eval_seeds = reference.params["eval_seeds"] if eval_seeds is None else int(eval_seeds)
    resolved_heuristic_max_order_size = (
        reference.heuristic_max_order_size if heuristic_max_order_size is None else int(heuristic_max_order_size)
    )
    args.eval_horizon = resolved_eval_horizon
    args.eval_seeds = resolved_eval_seeds
    args.max_order_size = resolved_heuristic_max_order_size
    args.reference_instance = reference_instance

    evaluations = evaluate_default_heuristics(args)
    literature_values = reference.literature_metadata["reported_values"]
    optimal_cost = literature_values.get("optimal")
    capped_base_stock_cost = literature_values.get("capped_base_stock")
    return {
        "reference_instance": reference.name,
        "description": reference.description,
        "params": reference.params,
        "literature_metadata": reference.literature_metadata,
        "evaluation_config": {
            "eval_horizon": resolved_eval_horizon,
            "eval_seeds": resolved_eval_seeds,
            "heuristic_max_order_size": resolved_heuristic_max_order_size,
        },
        "evaluation": evaluations,
        "optimal_reference": {
            "available": optimal_cost is not None and not (isinstance(optimal_cost, float) and np.isnan(optimal_cost)),
            "mean_cost": None
            if optimal_cost is None or (isinstance(optimal_cost, float) and np.isnan(optimal_cost))
            else float(optimal_cost),
            "source": "Xin2020TechnicalModels",
        },
        "capped_base_stock_reference": {
            "available": capped_base_stock_cost is not None
            and not (isinstance(capped_base_stock_cost, float) and np.isnan(capped_base_stock_cost)),
            "mean_cost": None
            if capped_base_stock_cost is None or (isinstance(capped_base_stock_cost, float) and np.isnan(capped_base_stock_cost))
            else float(capped_base_stock_cost),
            "source": "Xin2020TechnicalModels",
        },
        "ranking_check": {
            "myopic2_not_worse_than_myopic1": evaluations["myopic2"]["mean_cost"] <= evaluations["myopic1"]["mean_cost"] + 1e-9,
            "myopic2_not_worse_than_svbs": evaluations["myopic2"]["mean_cost"] <= evaluations["svbs"]["mean_cost"] + 1e-9,
        },
    }


def benchmark_grid(
    grid_name: str = "xin2020_extended_lost_sales",
    *,
    limit: int | None = None,
    **instance_overrides,
):
    grid = get_benchmark_grid(grid_name)
    instances = grid["instances"]
    if limit is not None:
        instances = instances[: int(limit)]

    results = [benchmark_reference_instance(instance["name"], **instance_overrides) for instance in instances]

    myopic2_beats_myopic1 = sum(result["ranking_check"]["myopic2_not_worse_than_myopic1"] for result in results)
    myopic2_beats_svbs = sum(result["ranking_check"]["myopic2_not_worse_than_svbs"] for result in results)
    return {
        "grid_name": grid["name"],
        "description": grid["description"],
        "num_instances": len(results),
        "grid_axes": grid["axes"],
        "results": results,
        "summary": {
            "myopic2_not_worse_than_myopic1_count": int(myopic2_beats_myopic1),
            "myopic2_not_worse_than_svbs_count": int(myopic2_beats_svbs),
        },
    }
