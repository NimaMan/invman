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
    evaluate_soft_tree_policy,
    get_exact_dp_summary,
    get_exact_verification_reference,
    soft_tree_rollout_kwargs,
)

import invman_rust


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Rust-backed soft-tree policy on the joint_replenishment exact verification instance."
    )
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
    parser.add_argument("--eval_seeds", type=int, default=2048)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _training_namespace(parsed, reference):
    run_tag = (
        f"joint_replenishment_verification_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_s{parsed.seed}_b{parsed.train_seed_batch}"
    )
    output_root = PACKAGE_ROOT / "outputs" / "joint_replenishment" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(reference["periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_model_fitness(model, reference):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            discounted_cost = invman_rust.joint_replenishment_soft_tree_rollout(
                seed=int(seed) + seed_offset,
                **soft_tree_rollout_kwargs(reference, model, flat_params=flat_params),
            )
            costs.append(float(discounted_cost))
        discounted_cost = float(np.mean(costs))
        reward = -discounted_cost
        if verbose:
            print(f"Seed {seed}: discounted cost {discounted_cost:.4f}, reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def _get_population_fitness(model, reference):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(params, dtype=np.float32).tolist() for params in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(reference, model, flat_params=model.get_model_flat_params()).items()
            if key != "flat_params"
        }
        batch_costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            batch_costs.append(
                invman_rust.joint_replenishment_soft_tree_population_rollout(
                    params_batch=params_batch,
                    seeds=[int(seed) + seed_offset for seed in seeds],
                    **rollout_kwargs,
                )
            )
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [(-float(discounted_cost), indiv_idx) for indiv_idx, discounted_cost in enumerate(costs.tolist())]

    return inner


def _comparison_table(exact_summary: dict, soft_tree_eval: dict, *, depth: int, leaf_type: str) -> list[dict]:
    rows = [
        {
            "policy": "optimal_dp",
            "params": "repo exact",
            "mean_cost": float(exact_summary["optimal_discounted_cost"]),
            "note": "exact benchmark",
        },
        {
            "policy": "minimum_order_quantity",
            "params": str(exact_summary["verification_reference"]["moq_item_targets"] + [
                exact_summary["verification_reference"]["moq_review_period"],
                exact_summary["verification_reference"]["moq_rounding_threshold"],
            ]),
            "mean_cost": float(exact_summary["moq_discounted_cost"]),
            "note": "benchmark heuristic",
        },
        {
            "policy": "dynamic_order_up_to",
            "params": str(exact_summary["verification_reference"]["dynout_item_targets"]),
            "mean_cost": float(exact_summary["dynout_discounted_cost"]),
            "note": "benchmark heuristic",
        },
        {
            "policy": "soft_tree",
            "params": f"d={depth}, leaf={leaf_type}",
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
        "| Policy | Params | Mean Discounted Cost | Gap vs Soft Tree Cost | Note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['params']}` | `{row['mean_cost']:.3f}` | `{row['gap_vs_soft_tree_cost']:.3f}` | {row['note']} |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    reference = get_exact_verification_reference()
    exact_summary = get_exact_dp_summary()
    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
    )

    train_args = _training_namespace(parsed, reference)
    trained_model, _ = train(
        model=model,
        get_model_fitness=_get_model_fitness(model, reference),
        get_population_fitness=_get_population_fitness(model, reference),
        args=train_args,
        same_seed=bool(parsed.same_seed),
    )

    eval_seeds = [parsed.seed + offset for offset in range(parsed.eval_seeds)]
    learned_evaluation = evaluate_soft_tree_policy(reference, trained_model, eval_seeds)
    comparison_rows = _comparison_table(
        exact_summary,
        learned_evaluation,
        depth=parsed.depth,
        leaf_type=parsed.leaf_type,
    )

    payload = {
        "reference": reference,
        "exact_summary": exact_summary,
        "tree_config": {
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

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["comparison_markdown"])


if __name__ == "__main__":
    main()
