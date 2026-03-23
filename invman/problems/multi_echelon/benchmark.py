from invman.problems.multi_echelon.heuristics import (
    evaluate_constant_base_stock_policy_across_seeds,
    search_best_constant_base_stock_policy,
)


def evaluate_default_heuristics(args):
    search_backend = "rust" if getattr(args, "rollout_backend", "python") == "rust" else "python"
    search_result = search_best_constant_base_stock_policy(
        args,
        seed=int(getattr(args, "seed", 1234)),
        horizon=min(int(args.horizon), 4000),
        backend=search_backend,
    )
    best_params = search_result["best_result"]["params"]
    eval_summary = evaluate_constant_base_stock_policy_across_seeds(
        args,
        params=best_params,
        num_seeds=int(getattr(args, "eval_seeds", 3)),
        horizon=int(getattr(args, "eval_horizon", args.horizon)),
    )
    eval_summary["search"] = search_result
    return {"constant_base_stock": eval_summary}
