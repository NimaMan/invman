from __future__ import annotations

from copy import copy

from invman.problems.dual_sourcing.dp import solve_bounded_dp
from invman.problems.dual_sourcing.heuristics import (
    evaluate_policy_across_seeds,
    search_best_capped_dual_index_policy,
    search_best_dual_index_policy,
    search_best_single_index_policy,
    search_best_tailored_base_surge_policy,
)


def evaluate_default_heuristics(args):
    search_backend = "rust" if getattr(args, "rollout_backend", "python") == "rust" else "python"
    search_horizon = min(int(args.horizon), 6000)
    results = {}
    searches = {
        "single_index": search_best_single_index_policy,
        "dual_index": search_best_dual_index_policy,
        "capped_dual_index": search_best_capped_dual_index_policy,
        "tailored_base_surge": search_best_tailored_base_surge_policy,
    }
    for name, search_fn in searches.items():
        search_summary = search_fn(args, seed=int(getattr(args, "seed", 1234)), horizon=search_horizon, backend=search_backend)
        eval_summary = evaluate_policy_across_seeds(
            args,
            policy_name=name,
            params=search_summary.best_result.params,
            num_seeds=int(getattr(args, "eval_seeds", 3)),
            horizon=int(getattr(args, "eval_horizon", args.horizon)),
        )
        eval_summary["search"] = search_summary.to_dict()
        results[name] = eval_summary
    return results


def benchmark_reference_instance(args, inventory_lower: int = -40, inventory_upper: int = 60):
    benchmark_args = copy(args)
    heuristic_results = evaluate_default_heuristics(benchmark_args)
    dp_result = solve_bounded_dp(
        benchmark_args,
        inventory_lower=inventory_lower,
        inventory_upper=inventory_upper,
    )
    return {
        "heuristics": heuristic_results,
        "bounded_dp": dp_result.to_dict(),
    }
