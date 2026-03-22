from __future__ import annotations

from copy import copy

import numpy as np

from invman.problems.lost_sales_fixed_order_cost.heuristics import (
    evaluate_policy_across_seeds,
    search_best_modified_s_s_q_policy,
    search_best_s_nq_policy,
    search_best_s_s_policy,
)
from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
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


def evaluate_default_heuristics(args):
    search_args = copy(args)
    search_args.horizon = args.horizon
    eval_args = copy(args)
    eval_args.horizon = args.eval_horizon

    s_s_summary = search_best_s_s_policy(
        args=search_args,
        seed=args.seed,
        horizon=args.horizon,
    )
    s_nq_summary = search_best_s_nq_policy(
        args=search_args,
        seed=args.seed,
        horizon=args.horizon,
    )
    modified_search = search_best_modified_s_s_q_policy(
        args=search_args,
        seed=args.seed,
        horizon=args.horizon,
        s_s_summary=s_s_summary,
    )

    return {
        "s_s": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_s",
            params=s_s_summary.best_result.params,
            num_seeds=args.eval_seeds,
            horizon=args.eval_horizon,
            track_demand=getattr(args, "track_demand", False),
        ),
        "s_nq": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_nq",
            params=s_nq_summary.best_result.params,
            num_seeds=args.eval_seeds,
            horizon=args.eval_horizon,
            track_demand=getattr(args, "track_demand", False),
        ),
        "modified_s_s_q": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="modified_s_s_q",
            params=modified_search["modified_policy"].best_result.params,
            num_seeds=args.eval_seeds,
            horizon=args.eval_horizon,
            track_demand=getattr(args, "track_demand", False),
        ),
    }


def benchmark_reference_instance(
    reference_instance: str,
    *,
    search_horizon: int | None = None,
    eval_horizon: int | None = None,
    eval_seeds: int | None = None,
    position_upper_bound: int | None = None,
    search_seed: int | None = None,
    top_k_s_s_pairs: int | None = None,
    q_window: int | None = None,
    backend: str = "python",
    modified_search_mode: str = "guided",
):
    reference = get_reference_instance(reference_instance)
    args = build_reference_args(reference_instance)

    resolved_search_horizon = reference["search"]["search_horizon"] if search_horizon is None else search_horizon
    resolved_eval_horizon = reference["evaluation"]["eval_horizon"] if eval_horizon is None else eval_horizon
    resolved_eval_seeds = reference["evaluation"]["eval_seeds"] if eval_seeds is None else eval_seeds
    resolved_position_upper_bound = (
        reference["search"]["position_upper_bound"]
        if position_upper_bound is None
        else position_upper_bound
    )
    resolved_search_seed = reference["search"]["search_seed"] if search_seed is None else search_seed
    resolved_top_k_s_s_pairs = (
        reference["search"]["top_k_s_s_pairs"] if top_k_s_s_pairs is None else top_k_s_s_pairs
    )
    resolved_q_window = reference["search"]["q_window"] if q_window is None else q_window

    search_args = copy(args)
    search_args.horizon = resolved_search_horizon
    eval_args = copy(args)
    eval_args.horizon = resolved_eval_horizon

    s_s_summary = search_best_s_s_policy(
        args=search_args,
        seed=resolved_search_seed,
        horizon=resolved_search_horizon,
        position_upper_bound=resolved_position_upper_bound,
        top_k=resolved_top_k_s_s_pairs,
        backend=backend,
    )
    s_nq_summary = search_best_s_nq_policy(
        args=search_args,
        seed=resolved_search_seed,
        horizon=resolved_search_horizon,
        position_upper_bound=resolved_position_upper_bound,
        backend=backend,
    )
    modified_search = search_best_modified_s_s_q_policy(
        args=search_args,
        seed=resolved_search_seed,
        horizon=resolved_search_horizon,
        position_upper_bound=resolved_position_upper_bound,
        top_k_s_s_pairs=resolved_top_k_s_s_pairs,
        q_window=resolved_q_window,
        s_s_summary=s_s_summary,
        search_mode=modified_search_mode,
        backend=backend,
    )

    evaluations = {
        "s_s": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_s",
            params=s_s_summary.best_result.params,
            num_seeds=resolved_eval_seeds,
            horizon=resolved_eval_horizon,
            track_demand=True,
        ),
        "s_nq": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_nq",
            params=s_nq_summary.best_result.params,
            num_seeds=resolved_eval_seeds,
            horizon=resolved_eval_horizon,
            track_demand=True,
        ),
        "modified_s_s_q": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="modified_s_s_q",
            params=modified_search["modified_policy"].best_result.params,
            num_seeds=resolved_eval_seeds,
            horizon=resolved_eval_horizon,
            track_demand=True,
        ),
    }

    return {
        "reference_instance": reference["name"],
        "description": reference["description"],
        "params": reference["params"],
        "literature_metadata": reference["literature_metadata"],
        "search_config": {
            "search_horizon": resolved_search_horizon,
            "position_upper_bound": resolved_position_upper_bound,
            "search_seed": resolved_search_seed,
            "top_k_s_s_pairs": resolved_top_k_s_s_pairs,
            "q_window": resolved_q_window,
            "backend": backend,
            "modified_search_mode": modified_search_mode,
        },
        "search_results": {
            "s_s": s_s_summary.to_dict(),
            "s_nq": s_nq_summary.to_dict(),
            "modified_s_s_q": modified_search["modified_policy"].to_dict(),
        },
        "evaluation": evaluations,
        "ranking_check": {
            "modified_not_worse_than_s_s": evaluations["modified_s_s_q"]["mean_cost"] <= evaluations["s_s"]["mean_cost"] + 1e-9,
            "modified_not_worse_than_s_nq": evaluations["modified_s_s_q"]["mean_cost"] <= evaluations["s_nq"]["mean_cost"] + 1e-9,
        },
    }


def benchmark_grid(
    grid_name: str = "literature_subset_poisson_mu5",
    *,
    limit: int | None = None,
    **instance_overrides,
):
    grid = get_benchmark_grid(grid_name)
    instances = grid["instances"]
    if limit is not None:
        instances = instances[: int(limit)]

    results = []
    for instance in instances:
        results.append(benchmark_reference_instance(instance["name"], **instance_overrides))

    modified_beats_s_s = sum(result["ranking_check"]["modified_not_worse_than_s_s"] for result in results)
    modified_beats_s_nq = sum(result["ranking_check"]["modified_not_worse_than_s_nq"] for result in results)
    avg_rel_improvement_vs_s_s = sum(
        100.0
        * (
            result["evaluation"]["s_s"]["mean_cost"] - result["evaluation"]["modified_s_s_q"]["mean_cost"]
        )
        / result["evaluation"]["s_s"]["mean_cost"]
        for result in results
    ) / len(results)
    avg_rel_improvement_vs_s_nq = sum(
        100.0
        * (
            result["evaluation"]["s_nq"]["mean_cost"] - result["evaluation"]["modified_s_s_q"]["mean_cost"]
        )
        / result["evaluation"]["s_nq"]["mean_cost"]
        for result in results
    ) / len(results)

    return {
        "grid_name": grid["name"],
        "description": grid["description"],
        "num_instances": len(results),
        "grid_axes": grid["axes"],
        "results": results,
        "summary": {
            "modified_not_worse_than_s_s_count": modified_beats_s_s,
            "modified_not_worse_than_s_nq_count": modified_beats_s_nq,
            "mean_relative_improvement_vs_s_s_pct": avg_rel_improvement_vs_s_s,
            "mean_relative_improvement_vs_s_nq_pct": avg_rel_improvement_vs_s_nq,
        },
    }
