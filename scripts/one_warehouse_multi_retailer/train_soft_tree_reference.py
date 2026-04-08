import argparse
import json
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
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_echelon_base_stock_policy,
    evaluate_soft_tree_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    get_primary_reference,
    get_reference,
    published_cost,
    search_best_echelon_base_stock,
    soft_tree_rollout_kwargs,
)

import invman_rust


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Rust-backed soft-tree policy on either the exact verifier or a literature-backed one_warehouse_multi_retailer reference instance."
    )
    parser.add_argument("--reference_name", default="exact")
    parser.add_argument("--train_allocation_policy", default="proportional")
    parser.add_argument("--eval_allocation_policy", default=None)
    parser.add_argument("--policy_action_mode", default="direct_orders")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--training_episodes", type=int, default=300)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--train_seed_batch", type=int, default=4)
    parser.add_argument("--eval_seeds", type=int, default=1024)
    parser.add_argument("--heuristic_search_replications", type=int, default=256)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _resolve_reference(name: str):
    if name == "exact":
        return get_exact_verification_reference(), "exact"
    if name == "primary":
        return get_primary_reference(), "literature"
    return get_reference(name), "literature"


def _training_namespace(parsed, reference: dict, reference_kind: str):
    ref_name = "verification" if reference_kind == "exact" else reference["name"]
    run_tag = (
        f"one_warehouse_multi_retailer_{ref_name}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_s{parsed.seed}_b{parsed.train_seed_batch}"
    )
    output_root = PACKAGE_ROOT / "outputs" / "one_warehouse_multi_retailer" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(reference["periods"] if reference_kind == "exact" else reference["benchmark_periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_model_fitness(model, reference, allocation_policy: str, policy_action_mode: str):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            discounted_cost = invman_rust.one_warehouse_multi_retailer_soft_tree_rollout(
                seed=int(seed) + seed_offset,
                **soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=flat_params,
                    allocation_policy=allocation_policy,
                    policy_action_mode=policy_action_mode,
                ),
            )
            costs.append(float(discounted_cost))
        discounted_cost = float(np.mean(costs))
        reward = -discounted_cost
        if verbose:
            print(f"Seed {seed}: cost {discounted_cost:.4f}, reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def _get_population_fitness(model, reference, allocation_policy: str, policy_action_mode: str):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(params, dtype=np.float32).tolist() for params in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=model.get_model_flat_params(),
                allocation_policy=allocation_policy,
                policy_action_mode=policy_action_mode,
            ).items()
            if key != "flat_params"
        }
        batch_costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            batch_costs.append(
                invman_rust.one_warehouse_multi_retailer_soft_tree_population_rollout(
                    params_batch=params_batch,
                    seeds=[int(seed) + seed_offset for seed in seeds],
                    **rollout_kwargs,
                )
            )
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [
            (-float(discounted_cost), indiv_idx)
            for indiv_idx, discounted_cost in enumerate(costs.tolist())
        ]

    return inner


def _comparison_rows_exact(exact_summary: dict, heuristics: dict, soft_tree_eval: dict, parsed) -> list[dict]:
    rows = [
        {
            "policy": "optimal_dp",
            "params": "repo exact",
            "mean_cost": float(exact_summary["optimal_discounted_cost"]),
            "note": "exact optimum",
        },
        {
            "policy": "best_echelon_base_stock_proportional",
            "params": str(
                [heuristics["proportional"]["warehouse_base_stock_level"]]
                + heuristics["proportional"]["retailer_base_stock_levels"]
            ),
            "mean_cost": float(heuristics["proportional"]["mean_cost"]),
            "note": "best exact heuristic",
        },
        {
            "policy": "best_echelon_base_stock_min_shortage",
            "params": str(
                [heuristics["min_shortage"]["warehouse_base_stock_level"]]
                + heuristics["min_shortage"]["retailer_base_stock_levels"]
            ),
            "mean_cost": float(heuristics["min_shortage"]["mean_cost"]),
            "note": "best exact heuristic",
        },
        {
            "policy": "soft_tree",
            "params": f"d={parsed.depth}, leaf={parsed.leaf_type}",
            "mean_cost": float(soft_tree_eval["mean_cost"]),
            "note": "trained policy",
        },
    ]
    learned_cost = float(soft_tree_eval["mean_cost"])
    for row in rows:
        row["gap_vs_soft_tree_cost"] = float(row["mean_cost"] - learned_cost)
    return rows


def _comparison_rows_literature(reference: dict, heuristics: dict, soft_tree_eval: dict, parsed) -> list[dict]:
    rows = [
        {
            "policy": "published_echelon_base_stock_proportional",
            "params": "paper",
            "mean_cost": float(published_cost(reference["published_proportional_benchmark"])),
            "note": "Kaynov Table A.3 reported cost",
        },
        {
            "policy": "published_echelon_base_stock_min_shortage",
            "params": "paper",
            "mean_cost": float(published_cost(reference["published_min_shortage_benchmark"])),
            "note": "Kaynov Table A.3 reported cost",
        },
        {
            "policy": "published_ppo",
            "params": "paper",
            "mean_cost": float(published_cost(reference["published_ppo_benchmark"])),
            "note": "Kaynov Table A.3 reported policy",
        },
        {
            "policy": "repo_echelon_base_stock_proportional",
            "params": str(
                [heuristics["proportional"]["warehouse_base_stock_level"]]
                + heuristics["proportional"]["retailer_base_stock_levels"]
            ),
            "mean_cost": float(heuristics["proportional"]["mean_cost"]),
            "note": "repo search/evaluation",
        },
        {
            "policy": "repo_echelon_base_stock_min_shortage",
            "params": str(
                [heuristics["min_shortage"]["warehouse_base_stock_level"]]
                + heuristics["min_shortage"]["retailer_base_stock_levels"]
            ),
            "mean_cost": float(heuristics["min_shortage"]["mean_cost"]),
            "note": "repo search/evaluation",
        },
        {
            "policy": "soft_tree",
            "params": f"d={parsed.depth}, leaf={parsed.leaf_type}",
            "mean_cost": float(soft_tree_eval["mean_cost"]),
            "note": "trained policy",
        },
    ]
    learned_cost = float(soft_tree_eval["mean_cost"])
    for row in rows:
        row["gap_vs_soft_tree_cost"] = float(row["mean_cost"] - learned_cost)
    return rows


def _markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Policy | Params | Mean Cost | Gap vs Soft Tree | Note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['params']}` | `{row['mean_cost']:.3f}` | `{row['gap_vs_soft_tree_cost']:.3f}` | {row['note']} |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    reference, reference_kind = _resolve_reference(parsed.reference_name)
    eval_allocation_policy = (
        parsed.train_allocation_policy
        if parsed.eval_allocation_policy is None
        else parsed.eval_allocation_policy
    )
    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
        policy_action_mode=parsed.policy_action_mode,
    )

    train_args = _training_namespace(parsed, reference, reference_kind)
    trained_model, _ = train(
            model=model,
        get_model_fitness=_get_model_fitness(
            model, reference, parsed.train_allocation_policy, parsed.policy_action_mode
        ),
        get_population_fitness=_get_population_fitness(
            model, reference, parsed.train_allocation_policy, parsed.policy_action_mode
        ),
        args=train_args,
        same_seed=bool(parsed.same_seed),
    )

    eval_seeds = [parsed.seed + offset for offset in range(parsed.eval_seeds)]
    learned_evaluation = evaluate_soft_tree_policy(
        reference,
        trained_model,
        eval_seeds,
        allocation_policy=eval_allocation_policy,
        policy_action_mode=parsed.policy_action_mode,
    )

    if reference_kind == "exact":
        exact_summary = get_exact_dp_summary()
        heuristics = {
            "proportional": search_best_echelon_base_stock(
                reference,
                allocation_policy="proportional",
            ),
            "min_shortage": search_best_echelon_base_stock(
                reference,
                allocation_policy="min_shortage",
            ),
        }
        comparison_rows = _comparison_rows_exact(
            exact_summary,
            heuristics,
            learned_evaluation,
            parsed,
        )
        payload = {
            "reference_kind": reference_kind,
            "reference": reference,
            "exact_summary": exact_summary,
            "heuristics": heuristics,
        }
    else:
        searched_heuristics = {
            "proportional": search_best_echelon_base_stock(
                reference,
                allocation_policy="proportional",
                replications=int(parsed.heuristic_search_replications),
                seed=int(parsed.seed),
            ),
            "min_shortage": search_best_echelon_base_stock(
                reference,
                allocation_policy="min_shortage",
                replications=int(parsed.heuristic_search_replications),
                seed=int(parsed.seed),
            ),
        }
        heuristics = {
            "proportional": evaluate_echelon_base_stock_policy(
                reference,
                warehouse_base_stock_level=searched_heuristics["proportional"]["warehouse_base_stock_level"],
                retailer_base_stock_levels=searched_heuristics["proportional"]["retailer_base_stock_levels"],
                allocation_policy="proportional",
                replications=int(parsed.eval_seeds),
                seed=int(parsed.seed),
            ),
            "min_shortage": evaluate_echelon_base_stock_policy(
                reference,
                warehouse_base_stock_level=searched_heuristics["min_shortage"]["warehouse_base_stock_level"],
                retailer_base_stock_levels=searched_heuristics["min_shortage"]["retailer_base_stock_levels"],
                allocation_policy="min_shortage",
                replications=int(parsed.eval_seeds),
                seed=int(parsed.seed),
            ),
        }
        comparison_rows = _comparison_rows_literature(
            reference,
            heuristics,
            learned_evaluation,
            parsed,
        )
        payload = {
            "reference_kind": reference_kind,
            "reference": reference,
            "heuristics": heuristics,
            "heuristic_search_results": searched_heuristics,
        }

    payload.update(
        {
            "tree_config": {
                "train_allocation_policy": parsed.train_allocation_policy,
                "eval_allocation_policy": eval_allocation_policy,
                "policy_action_mode": parsed.policy_action_mode,
                "depth": parsed.depth,
                "temperature": parsed.temperature,
                "split_type": parsed.split_type,
                "leaf_type": parsed.leaf_type,
                "training_episodes": parsed.training_episodes,
                "es_population": parsed.es_population,
                "sigma_init": parsed.sigma_init,
                "seed": parsed.seed,
                "same_seed": parsed.same_seed,
                "train_seed_batch": parsed.train_seed_batch,
            },
            "evaluation": {
                "soft_tree": learned_evaluation,
            },
            "comparison_rows": comparison_rows,
            "comparison_markdown": _markdown_table(comparison_rows),
        }
    )

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["comparison_markdown"])


if __name__ == "__main__":
    main()
