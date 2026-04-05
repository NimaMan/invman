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
    MEDIUM_REFERENCE_INSTANCE_NAME,
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_heuristic_policy,
    evaluate_soft_tree_policy,
    get_reference,
    search_best_base_stock,
    search_best_bsp_low_ew,
    soft_tree_rollout_kwargs,
)

import invman_rust


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Rust-backed soft-tree policy on a perishable reference instance.")
    parser.add_argument("--reference", default=MEDIUM_REFERENCE_INSTANCE_NAME)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--training_episodes", type=int, default=300)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--eval_seeds", type=int, default=128)
    parser.add_argument("--heuristic_search_seeds", type=int, default=32)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _training_namespace(parsed, reference):
    run_tag = f"perishable_{parsed.reference}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
    output_root = PACKAGE_ROOT / "outputs" / "perishable_inventory" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(reference["horizon"]),
        seed=int(parsed.seed),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_model_fitness(model, reference):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        discounted_return = invman_rust.perishable_inventory_soft_tree_discounted_return(
            seed=int(seed),
            **soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=flat_params,
                horizon=int(args.horizon),
            ),
        )
        if verbose:
            print(f"Seed {seed}: discounted return {discounted_return:.4f}")
        return float(discounted_return), indiv_idx

    return inner


def _get_population_fitness(model, reference):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(params, dtype=np.float32).tolist() for params in model_params_batch]
        returns = invman_rust.perishable_inventory_soft_tree_population_discounted_return(
            params_batch=params_batch,
            seeds=[int(seed) for seed in seeds],
            **{
                key: value
                for key, value in soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=model.get_model_flat_params(),
                    horizon=int(args.horizon),
                ).items()
                if key != "flat_params"
            },
        )
        return [(float(discounted_return), indiv_idx) for indiv_idx, discounted_return in enumerate(returns)]

    return inner


def _comparison_table(reference, learned_summary, base_stock_eval, bsp_low_ew_eval):
    published = reference["published_scenario_a_returns"]
    rows = [
        {
            "policy": "published_value_iteration",
            "params": "-",
            "mean_return": float(published["value_iteration_mean_return"]),
            "note": "paper benchmark",
        },
        {
            "policy": "published_base_stock",
            "params": "-",
            "mean_return": float(published["best_base_stock_mean_return"]),
            "note": "paper benchmark",
        },
        {
            "policy": "repo_base_stock",
            "params": str(base_stock_eval["params"]),
            "mean_return": float(base_stock_eval["mean_return"]),
            "note": "Rust heuristic on eval seeds",
        },
        {
            "policy": "repo_bsp_low_ew",
            "params": str(bsp_low_ew_eval["params"]),
            "mean_return": float(bsp_low_ew_eval["mean_return"]),
            "note": "Rust heuristic on eval seeds",
        },
        {
            "policy": "soft_tree",
            "params": f"d={learned_summary['depth']}, leaf={learned_summary['leaf_type']}",
            "mean_return": float(learned_summary["evaluation"]["mean_return"]),
            "note": "trained policy",
        },
    ]
    learned_return = float(learned_summary["evaluation"]["mean_return"])
    for row in rows:
        row["gap_vs_soft_tree_return"] = float(row["mean_return"] - learned_return)
    return rows


def _markdown_table(rows):
    lines = [
        "| Policy | Params | Mean Return | Gap vs Soft Tree Return | Note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['params']}` | `{row['mean_return']:.3f}` | "
            f"`{row['gap_vs_soft_tree_return']:.3f}` | {row['note']} |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    reference = get_reference(parsed.reference)
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
    heuristic_search_seeds = [parsed.seed + offset for offset in range(parsed.heuristic_search_seeds)]

    learned_evaluation = evaluate_soft_tree_policy(reference, trained_model, eval_seeds)
    base_stock_search = search_best_base_stock(reference, heuristic_search_seeds)
    bsp_low_ew_search = search_best_bsp_low_ew(reference, heuristic_search_seeds)
    base_stock_eval = evaluate_heuristic_policy(
        reference,
        "base_stock",
        tuple(base_stock_search["best"]["params"]),
        eval_seeds,
    )
    bsp_low_ew_eval = evaluate_heuristic_policy(
        reference,
        "bsp_low_ew",
        tuple(bsp_low_ew_search["best"]["params"]),
        eval_seeds,
    )

    comparison_rows = _comparison_table(
        reference,
        {
            "depth": parsed.depth,
            "leaf_type": parsed.leaf_type,
            "evaluation": learned_evaluation,
        },
        base_stock_eval,
        bsp_low_ew_eval,
    )

    payload = {
        "reference_instance": parsed.reference,
        "reference": reference,
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
        },
        "evaluation": {
            "soft_tree": learned_evaluation,
            "base_stock": {
                "search": base_stock_search,
                "eval": base_stock_eval,
            },
            "bsp_low_ew": {
                "search": bsp_low_ew_search,
                "eval": bsp_low_ew_eval,
            },
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
