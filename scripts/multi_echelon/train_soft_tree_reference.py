import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.es_mp import train

from common import (
    action_grids,
    benchmark_allocation_mode,
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_stationary_policy,
    evaluate_soft_tree_policy,
    exact_evaluate_soft_tree,
    get_exact_dp_summary,
    get_exact_verification_reference,
    get_primary_reference,
    get_reference,
    is_exact_reference,
    initialize_soft_tree_to_constant_action,
    savings_pct,
    search_best_constant_base_stock,
    soft_tree_rollout_kwargs,
)

import invman_rust


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Rust-backed soft-tree policy on either the exact multi_echelon verifier or one of the Gijs / Van Roy literature settings."
    )
    parser.add_argument("--reference_name", default="primary")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--include_period_feature", action="store_true")
    parser.add_argument(
        "--policy_feature_mode",
        choices=[
            "full_decision_state",
            "symmetric_summary",
            "compact_summary",
        ],
        default="full_decision_state",
    )
    parser.add_argument(
        "--policy_action_mode",
        choices=["direct_base_stock", "anchor_adjustment", "direct_warehouse_order_store_target"],
        default="anchor_adjustment",
    )
    parser.add_argument("--training_horizon", type=int, default=None)
    parser.add_argument("--training_episodes", type=int, default=300)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--train_seed_batch", type=int, default=8)
    parser.add_argument("--eval_seeds", type=int, default=256)
    parser.add_argument("--eval_seed_start", type=int, default=1_000_000)
    parser.add_argument("--heuristic_search_replications", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def resolve_reference(name: str) -> dict:
    if name == "exact":
        return get_exact_verification_reference()
    if name == "primary":
        return get_primary_reference()
    return get_reference(name)


def training_namespace(parsed, reference: dict) -> SimpleNamespace:
    ref_name = "exact" if is_exact_reference(reference) else reference["name"]
    temp_tag = f"{int(round(parsed.temperature * 1000)):03d}"
    feature_tag = "full" if is_exact_reference(reference) else str(parsed.policy_feature_mode)
    action_tag = "direct" if is_exact_reference(reference) else str(parsed.policy_action_mode)
    run_tag = (
        f"multi_echelon_{ref_name}_{feature_tag}_{action_tag}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_t{temp_tag}_s{parsed.seed}_b{parsed.train_seed_batch}"
    )
    output_root = PACKAGE_ROOT / "outputs" / "multi_echelon" / run_tag
    horizon = int(parsed.training_horizon or (reference["periods"] if is_exact_reference(reference) else min(reference["benchmark_periods"], 10_000)))
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(horizon),
        seed=int(parsed.seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def get_model_fitness(model, reference: dict, include_period_feature: bool, training_horizon: int):
    rollout_overrides = {}
    if not is_exact_reference(reference):
        rollout_overrides = dict(getattr(model, "_multi_echelon_rollout_overrides", {}))

    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        if is_exact_reference(reference):
            evaluation = exact_evaluate_soft_tree(
                reference,
                model,
                flat_params=flat_params,
                include_period_feature=include_period_feature,
            )
            mean_cost = float(evaluation["discounted_cost"])
        else:
            rollout_kwargs = soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=flat_params,
                include_period_feature=include_period_feature,
                **rollout_overrides,
            )
            rollout_kwargs["horizon"] = int(training_horizon)
            for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
                discounted_cost = invman_rust.multi_echelon_soft_tree_rollout(
                    seed=int(seed) + seed_offset,
                    **rollout_kwargs,
                )
                costs.append(float(discounted_cost))
            mean_cost = float(np.mean(costs))
        reward = -mean_cost
        if verbose:
            print(f"Seed {seed}: cost {mean_cost:.4f}, reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def get_population_fitness(model, reference: dict, include_period_feature: bool, training_horizon: int):
    rollout_overrides = {}
    if not is_exact_reference(reference):
        rollout_overrides = dict(getattr(model, "_multi_echelon_rollout_overrides", {}))

    def inner(_model, args, model_params_batch, seeds):
        del _model
        if is_exact_reference(reference):
            values = []
            for idx, params in enumerate(model_params_batch):
                evaluation = exact_evaluate_soft_tree(
                    reference,
                    model,
                    flat_params=params,
                    include_period_feature=include_period_feature,
                )
                values.append((-float(evaluation["discounted_cost"]), idx))
            return values
        params_batch = [np.asarray(params, dtype=np.float32).tolist() for params in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=model.get_model_flat_params(),
                include_period_feature=include_period_feature,
                **rollout_overrides,
            ).items()
            if key != "flat_params"
        }
        rollout_kwargs["horizon"] = int(training_horizon)
        batch_costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            batch_costs.append(
                invman_rust.multi_echelon_soft_tree_population_rollout(
                    params_batch=params_batch,
                    seeds=[int(seed) + seed_offset for seed in seeds],
                    **rollout_kwargs,
                )
            )
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [(-float(cost), idx) for idx, cost in enumerate(costs.tolist())]

    return inner


def comparison_rows_exact(reference: dict, exact_summary: dict, exact_tree: dict) -> list[dict]:
    rows = [
        {
            "policy": "optimal_dp",
            "params": "repo exact",
            "mean_cost": float(exact_summary["optimal_discounted_cost"]),
            "note": "exact optimum",
        },
        {
            "policy": "constant_base_stock_sequential",
            "params": str(exact_summary["sequential_levels"]),
            "mean_cost": float(exact_summary["sequential_discounted_cost"]),
            "note": "exact stationary heuristic",
        },
        {
            "policy": "constant_base_stock_proportional",
            "params": str(exact_summary["proportional_levels"]),
            "mean_cost": float(exact_summary["proportional_discounted_cost"]),
            "note": "exact stationary heuristic",
        },
        {
            "policy": "constant_base_stock_min_shortage",
            "params": str(exact_summary["min_shortage_levels"]),
            "mean_cost": float(exact_summary["min_shortage_discounted_cost"]),
            "note": "literature-style exact heuristic",
        },
        {
            "policy": "soft_tree",
            "params": f"d=2 exact soft tree",
            "mean_cost": float(exact_tree["discounted_cost"]),
            "note": "trained policy",
        },
    ]
    learned_cost = float(exact_tree["discounted_cost"])
    for row in rows:
        row["gap_vs_soft_tree_cost"] = float(row["mean_cost"] - learned_cost)
    return rows


def comparison_rows_literature(
    reference: dict,
    repo_grid_heuristic: dict,
    repo_published_row: dict | None,
    learned: dict,
) -> list[dict]:
    learned_cost = float(learned["mean_cost"])
    published_constant_cost = reference.get("published_constant_base_stock_mean_cost")
    rows = []

    if published_constant_cost is not None:
        rows.append(
            {
                "policy": "published_constant_base_stock",
                "params": str([int(value) for value in reference["published_constant_base_stock_levels"]]),
                "mean_cost": float(published_constant_cost),
                "note": "paper row",
            }
        )
    if repo_published_row is not None:
        rows.append(
            {
                "policy": "repo_constant_base_stock_at_published_levels",
                "params": str(
                    [int(repo_published_row["warehouse_level"]), int(repo_published_row["retailer_level"])]
                ),
                "mean_cost": float(repo_published_row["mean_cost"]),
                "note": "repo reproduction at published heuristic levels",
            }
        )

    rows.append(
        {
            "policy": "repo_grid_constant_base_stock",
            "params": str([repo_grid_heuristic["warehouse_level"], repo_grid_heuristic["retailer_level"]]),
            "mean_cost": float(repo_grid_heuristic["mean_cost"]),
            "note": "best reproduced constant base-stock over carried search grid",
        }
    )

    if reference.get("published_van_roy_best_ndp_mean_cost") is not None:
        rows.append(
            {
                "policy": "published_van_roy_ndp",
                "params": "paper",
                "mean_cost": float(reference["published_van_roy_best_ndp_mean_cost"]),
                "note": "paper row",
            }
        )
    if reference.get("published_a3c_savings_pct") is not None and published_constant_cost is not None:
        target_cost = float(published_constant_cost) * (
            1.0 - float(reference["published_a3c_savings_pct"]) / 100.0
        )
        rows.append(
            {
                "policy": "published_a3c_implied_cost",
                "params": "paper relative row",
                "mean_cost": target_cost,
                "note": "published A3C savings applied to the published constant base-stock cost",
            }
        )

    rows.append(
        {
            "policy": "soft_tree",
            "params": f"d={learned['depth']}, {learned['split_type']}, {learned['leaf_type']}",
            "mean_cost": learned_cost,
            "note": "trained policy",
        }
    )

    for row in rows:
        row["gap_vs_soft_tree_cost"] = float(row["mean_cost"] - learned_cost)
        row["savings_vs_published_constant_pct"] = (
            savings_pct(float(published_constant_cost), float(row["mean_cost"]))
            if published_constant_cost is not None
            else float("nan")
        )
    return rows


def markdown_table_exact(rows: list[dict]) -> str:
    lines = [
        "| Policy | Params | Exact Discounted Cost | Gap vs Soft Tree | Note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['params']}` | `{row['mean_cost']:.6f}` | `{row['gap_vs_soft_tree_cost']:.6f}` | {row['note']} |"
        )
    return "\n".join(lines)


def markdown_table_literature(rows: list[dict]) -> str:
    lines = [
        "| Policy | Params | Mean Cost | Gap vs Soft Tree | Savings vs Published Constant | Note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        mean_cost = "nan" if math.isnan(row["mean_cost"]) else f"{row['mean_cost']:.6f}"
        savings = (
            "nan"
            if math.isnan(row["savings_vs_published_constant_pct"])
            else f"{row['savings_vs_published_constant_pct']:.3f}%"
        )
        lines.append(
            f"| `{row['policy']}` | `{row['params']}` | `{mean_cost}` | `{row['gap_vs_soft_tree_cost']:.6f}` | `{savings}` | {row['note']} |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    reference = resolve_reference(parsed.reference_name)
    include_period_feature = bool(parsed.include_period_feature or is_exact_reference(reference))
    initialization = None
    model_kwargs = {}
    rollout_overrides = {}
    init_action = None

    if is_exact_reference(reference):
        exact_summary = get_exact_dp_summary()
        heuristic_candidates = [
            (
                "sequential",
                list(exact_summary["sequential_levels"]),
                float(exact_summary["sequential_discounted_cost"]),
            ),
            (
                "proportional",
                list(exact_summary["proportional_levels"]),
                float(exact_summary["proportional_discounted_cost"]),
            ),
            (
                "min_shortage",
                list(exact_summary["min_shortage_levels"]),
                float(exact_summary["min_shortage_discounted_cost"]),
            ),
        ]
        best_name, best_levels, best_cost = min(heuristic_candidates, key=lambda row: row[2])
        initialization = {
            "kind": "best_exact_stationary_heuristic",
            "policy": best_name,
            "levels": [int(value) for value in best_levels],
            "reference_cost": float(best_cost),
        }
        model_kwargs["policy_feature_mode"] = "full_decision_state"
        init_action = initialization["levels"]
    else:
        init_search_replications = int(
            parsed.heuristic_search_replications or min(int(reference["benchmark_replications"]), 16)
        )
        init_search = search_best_constant_base_stock(
            reference,
            allocation_mode=benchmark_allocation_mode(reference),
            replications=init_search_replications,
            seed=int(parsed.seed),
            top_k=10,
        )
        init_best = dict(init_search["best_result"])
        initialization = {
            "kind": "best_constant_base_stock",
            "levels": [int(init_best["warehouse_level"]), int(init_best["retailer_level"])],
            "search_replications": int(init_search_replications),
            "search_mean_cost": float(init_best["mean_cost"]),
        }
        policy_feature_mode = str(parsed.policy_feature_mode)
        policy_action_mode = str(parsed.policy_action_mode)
        benchmark_warehouse_levels, benchmark_retailer_levels = action_grids(reference)
        warehouse_anchor_level, retailer_anchor_level = initialization["levels"]
        if (
            policy_action_mode == "direct_warehouse_order_store_target"
            and reference.get("published_constant_base_stock_levels")
        ):
            retailer_anchor_level = int(reference["published_constant_base_stock_levels"][1])
        warehouse_adjustments = sorted(
            {
                int(warehouse_anchor_level - level)
                for level in benchmark_warehouse_levels
                if int(level) <= int(warehouse_anchor_level)
            }
        )
        retailer_adjustments = sorted(
            {
                int(level - retailer_anchor_level)
                for level in benchmark_retailer_levels
                if int(level) >= int(retailer_anchor_level)
            }
        )
        model_kwargs = {
            "policy_feature_mode": policy_feature_mode,
        }
        rollout_overrides = {
            "policy_feature_mode": policy_feature_mode,
            "policy_action_mode": policy_action_mode,
            "warehouse_anchor_level": int(warehouse_anchor_level),
            "retailer_anchor_level": int(retailer_anchor_level),
            "reference_warehouse_levels": benchmark_warehouse_levels,
            "reference_retailer_levels": benchmark_retailer_levels,
        }
        if policy_action_mode == "anchor_adjustment":
            model_kwargs["warehouse_levels"] = warehouse_adjustments
            model_kwargs["retailer_levels"] = retailer_adjustments
            initialization["adjustment_levels"] = {
                "warehouse": warehouse_adjustments,
                "retailer": retailer_adjustments,
            }
            init_action = [0, 0]
        elif policy_action_mode == "direct_warehouse_order_store_target":
            init_action = [
                int(benchmark_warehouse_levels[len(benchmark_warehouse_levels) // 2]),
                int(retailer_anchor_level),
            ]
        else:
            init_action = initialization["levels"]
        initialization["policy_feature_mode"] = policy_feature_mode
        initialization["policy_action_mode"] = policy_action_mode

    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
        include_period_feature=include_period_feature,
        **model_kwargs,
    )
    initialize_soft_tree_to_constant_action(
        model,
        init_action,
    )
    model._multi_echelon_rollout_overrides = dict(rollout_overrides)
    train_args = training_namespace(parsed, reference)
    training_horizon = int(train_args.horizon)
    trained_model, _ = train(
        model=model,
        get_model_fitness=get_model_fitness(
            model, reference, include_period_feature, training_horizon
        ),
        get_population_fitness=get_population_fitness(
            model, reference, include_period_feature, training_horizon
        ),
        args=train_args,
        same_seed=bool(parsed.same_seed),
    )

    eval_seeds = [parsed.eval_seed_start + offset for offset in range(parsed.eval_seeds)]

    if is_exact_reference(reference):
        exact_tree = exact_evaluate_soft_tree(
            reference,
            trained_model,
            include_period_feature=include_period_feature,
        )
        mc_tree = evaluate_soft_tree_policy(
            reference,
            trained_model,
            eval_seeds,
            include_period_feature=include_period_feature,
        )
        rows = comparison_rows_exact(reference, exact_summary, exact_tree)
        payload = {
            "reference_kind": "exact",
            "reference": reference,
            "initialization": initialization,
            "exact_summary": exact_summary,
            "soft_tree_exact": exact_tree,
            "soft_tree_monte_carlo": mc_tree,
            "comparison_rows": rows,
            "markdown": markdown_table_exact(rows),
        }
    else:
        search_replications = int(
            parsed.heuristic_search_replications or min(int(reference["benchmark_replications"]), 16)
        )
        heuristic_search = search_best_constant_base_stock(
            reference,
            allocation_mode=benchmark_allocation_mode(reference),
            replications=search_replications,
            seed=int(parsed.seed),
            top_k=10,
        )
        heuristic_best = dict(heuristic_search["best_result"])
        heuristic_costs = []
        for seed in eval_seeds:
            heuristic_costs.append(
                float(
                    invman_rust.multi_echelon_search_stationary_policy(
                        policy_kind="constant_base_stock",
                        allocation_mode=str(benchmark_allocation_mode(reference)),
                        warehouse_levels=[int(heuristic_best["warehouse_level"])],
                        retailer_levels=[int(heuristic_best["retailer_level"])],
                        warehouse_lead_time=int(reference["warehouse_lead_time"]),
                        retailer_lead_time=int(reference["retailer_lead_time"]),
                        num_retailers=int(reference["num_retailers"]),
                        warehouse_holding_cost=float(reference["warehouse_holding_cost"]),
                        retailer_holding_cost=float(reference["retailer_holding_cost"]),
                        warehouse_expedited_cost=float(reference["warehouse_expedited_cost"]),
                        warehouse_lost_sale_cost=float(reference["warehouse_lost_sale_cost"]),
                        expedited_service_prob=float(reference["expedited_service_prob"]),
                        warehouse_capacity=int(reference["warehouse_capacity"]),
                        warehouse_inventory_cap=int(reference["warehouse_inventory_cap"]),
                        retailer_inventory_cap=int(reference["retailer_inventory_cap"]),
                        inventory_dynamics_mode=str(reference["inventory_dynamics_mode"]),
                        demand_distribution=str(reference["demand_distribution"]),
                        demand_mean=float(reference["demand_mean"]),
                        demand_std=float(reference["demand_std"]),
                        horizon=int(reference["benchmark_periods"]),
                        replications=1,
                        seed=int(seed),
                        warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
                        discount_factor=1.0,
                        objective=str(reference["rollout_objective"]),
                        top_k=1,
                    )["best_result"]["mean_cost"]
                )
            )
        heuristic_eval = {
            "warehouse_level": int(heuristic_best["warehouse_level"]),
            "retailer_level": int(heuristic_best["retailer_level"]),
            "mean_cost": float(np.mean(heuristic_costs)),
            "cost_std": float(np.std(heuristic_costs)),
            "num_samples": int(len(heuristic_costs)),
            "costs": heuristic_costs,
        }
        repo_published_row = None
        if reference.get("published_constant_base_stock_mean_cost") is not None:
            published_levels = [int(value) for value in reference["published_constant_base_stock_levels"]]
            repo_published_row = evaluate_stationary_policy(
                reference,
                warehouse_level=published_levels[0],
                retailer_level=published_levels[1],
                allocation_mode=benchmark_allocation_mode(reference),
                policy_kind="regular_base_stock",
                replications=int(parsed.eval_seeds),
                seed=int(parsed.eval_seed_start),
            )
        learned_eval = evaluate_soft_tree_policy(
            reference,
            trained_model,
            eval_seeds,
            include_period_feature=include_period_feature,
            **rollout_overrides,
        )
        learned_eval.update(
            {
                "depth": int(parsed.depth),
                "split_type": parsed.split_type,
                "leaf_type": parsed.leaf_type,
            }
        )
        rows = comparison_rows_literature(reference, heuristic_eval, repo_published_row, learned_eval)
        payload = {
            "reference_kind": "literature",
            "reference": reference,
            "initialization": initialization,
            "heuristic_search": heuristic_search,
            "heuristic_best": heuristic_best,
            "repo_grid_constant_base_stock_evaluation": heuristic_eval,
            "repo_published_constant_base_stock_evaluation": repo_published_row,
            "soft_tree_evaluation": learned_eval,
            "repo_soft_tree_savings_pct_vs_grid_base": savings_pct(
                heuristic_eval["mean_cost"], learned_eval["mean_cost"]
            ),
            "soft_tree_savings_pct_vs_published_constant": (
                savings_pct(
                    float(reference["published_constant_base_stock_mean_cost"]),
                    learned_eval["mean_cost"],
                )
                if reference.get("published_constant_base_stock_mean_cost") is not None
                else float("nan")
            ),
            "published_constant_base_stock_mean_cost": reference.get(
                "published_constant_base_stock_mean_cost"
            ),
            "published_van_roy_best_ndp_mean_cost": reference.get(
                "published_van_roy_best_ndp_mean_cost"
            ),
            "published_a3c_savings_pct": reference.get("published_a3c_savings_pct"),
            "published_a3c_confidence_half_width_pct": reference.get(
                "published_a3c_confidence_half_width_pct"
            ),
            "published_van_roy_savings_pct_approx": reference.get(
                "published_van_roy_savings_pct_approx"
            ),
            "comparison_rows": rows,
            "markdown": markdown_table_literature(rows),
        }

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
