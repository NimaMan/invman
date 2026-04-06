from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.benchmarks.practical import load_dataset
from invman.es_mp import train

from common import (
    MEDIUM_REFERENCE_INSTANCE_NAME,
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_heuristic_policy,
    evaluate_heuristic_trace_summary,
    evaluate_soft_tree_policy,
    evaluate_soft_tree_trace_summary,
    get_reference,
    search_best_base_stock,
    search_best_base_stock_from_demands,
    search_best_bsp_low_ew,
    search_best_bsp_low_ew_from_demands,
    soft_tree_rollout_kwargs,
)

import invman_rust


DEFAULT_EXACT_REFERENCES = [
    "de_moor2022_m2_exp1_l1_cp7_lifo",
    "de_moor2022_m2_exp2_l1_cp7_fifo",
]

DEFAULT_PRACTICAL_DATASET = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "practical"
    / "datasets"
    / "grocery_like_daily_trace.json"
)

DEFAULT_OUTPUT_JSON = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "experiments"
    / "reports"
    / "latest_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "experiments"
    / "reports"
    / "README.md"
)

METRIC_ORDER = [
    "mean_period_cost",
    "fill_rate",
    "cycle_service_level",
    "waste_rate",
    "mean_holding_inventory",
]

METRIC_LABELS = {
    "mean_period_cost": "Mean Period Cost",
    "fill_rate": "Fill Rate",
    "cycle_service_level": "Cycle Service",
    "waste_rate": "Waste / Demand",
    "mean_holding_inventory": "Mean Holding",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the perishable-inventory paper benchmark: CMA-ES soft-tree vs heuristics vs exact optimum."
    )
    parser.add_argument("--exact_references", nargs="+", default=list(DEFAULT_EXACT_REFERENCES))
    parser.add_argument("--practical_dataset", default=str(DEFAULT_PRACTICAL_DATASET))
    parser.add_argument("--practical_reference", default=MEDIUM_REFERENCE_INSTANCE_NAME)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--leaf_types", nargs="+", default=["linear", "sigmoid_linear"])
    parser.add_argument("--exact_training_episodes", type=int, default=160)
    parser.add_argument("--practical_training_episodes", type=int, default=240)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--eval_seeds", type=int, default=128)
    parser.add_argument("--heuristic_search_seeds", type=int, default=32)
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser.parse_args()


def empirical_mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def stable_seed(base_seed: int, tag: str) -> int:
    return int(base_seed + (zlib.adler32(tag.encode("utf-8")) % 1_000_000))


def _training_namespace(run_tag: str, *, horizon: int, training_episodes: int, es_population: int, sigma_init: float, seed: int):
    output_root = PACKAGE_ROOT / "outputs" / "perishable_inventory" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(sigma_init),
        es_population=int(es_population),
        training_episodes=int(training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(training_episodes)),
        save_solutions=False,
        horizon=int(horizon),
        seed=int(seed),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _stochastic_model_fitness(model, reference):
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


def _stochastic_population_fitness(model, reference):
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


def _deterministic_trace_model_fitness(model, reference, demands: list[int], demand_mean: float):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, seed, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        mean_period_cost = invman_rust.perishable_inventory_soft_tree_rollout_from_demands(
            on_hand=[0 for _ in range(int(reference["shelf_life"]))],
            pipeline_orders=[0 for _ in range(max(int(reference["lead_time"]) - 1, 0))],
            demands=[int(value) for value in demands],
            demand_mean=float(demand_mean),
            warm_up_periods_ratio=0.0,
            **{
                key: value
                for key, value in soft_tree_rollout_kwargs(
                    reference,
                    model,
                    flat_params=flat_params,
                    horizon=int(args.horizon),
                ).items()
                if key
                not in {
                    "flat_params",
                    "demand_mean",
                    "demand_cov",
                    "shelf_life",
                    "lead_time",
                    "horizon",
                    "warm_up_periods_ratio",
                }
            },
        )
        reward = -float(mean_period_cost)
        if verbose:
            print(f"Train-trace reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def _deterministic_trace_population_fitness(model, reference, demands: list[int], demand_mean: float):
    def inner(_model, args, model_params_batch, seeds):
        del _model, seeds
        results = []
        for indiv_idx, params in enumerate(model_params_batch):
            mean_period_cost = invman_rust.perishable_inventory_soft_tree_rollout_from_demands(
                flat_params=np.asarray(params, dtype=np.float32).tolist(),
                on_hand=[0 for _ in range(int(reference["shelf_life"]))],
                pipeline_orders=[0 for _ in range(max(int(reference["lead_time"]) - 1, 0))],
                demands=[int(value) for value in demands],
                demand_mean=float(demand_mean),
                warm_up_periods_ratio=0.0,
                **{
                    key: value
                    for key, value in soft_tree_rollout_kwargs(
                        reference,
                        model,
                        flat_params=model.get_model_flat_params(),
                        horizon=int(args.horizon),
                    ).items()
                    if key
                    not in {
                        "flat_params",
                        "demand_mean",
                        "demand_cov",
                        "shelf_life",
                        "lead_time",
                        "horizon",
                        "warm_up_periods_ratio",
                    }
                },
            )
            results.append((-float(mean_period_cost), indiv_idx))
        return results

    return inner


def train_soft_tree_on_reference(
    reference: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    training_episodes: int,
    es_population: int,
    sigma_init: float,
    seed: int,
    same_seed: bool,
    run_tag: str,
):
    model = build_soft_tree_model(
        reference,
        depth=depth,
        temperature=temperature,
        split_type=split_type,
        leaf_type=leaf_type,
    )
    train_args = _training_namespace(
        run_tag,
        horizon=int(reference["horizon"]),
        training_episodes=training_episodes,
        es_population=es_population,
        sigma_init=sigma_init,
        seed=seed,
    )
    trained_model, _ = train(
        model=model,
        get_model_fitness=_stochastic_model_fitness(model, reference),
        get_population_fitness=_stochastic_population_fitness(model, reference),
        args=train_args,
        same_seed=bool(same_seed),
    )
    return trained_model


def train_soft_tree_on_demand_trace(
    reference: dict,
    demands: list[int],
    *,
    demand_mean: float,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    training_episodes: int,
    es_population: int,
    sigma_init: float,
    seed: int,
    same_seed: bool,
    run_tag: str,
):
    model = build_soft_tree_model(
        reference,
        depth=depth,
        temperature=temperature,
        split_type=split_type,
        leaf_type=leaf_type,
    )
    train_args = _training_namespace(
        run_tag,
        horizon=len(demands),
        training_episodes=training_episodes,
        es_population=es_population,
        sigma_init=sigma_init,
        seed=seed,
    )
    trained_model, _ = train(
        model=model,
        get_model_fitness=_deterministic_trace_model_fitness(model, reference, demands, demand_mean),
        get_population_fitness=_deterministic_trace_population_fitness(model, reference, demands, demand_mean),
        args=train_args,
        same_seed=bool(same_seed),
    )
    return trained_model


def build_exact_slice_rows(reference_name: str, parsed, eval_seeds: list[int], heuristic_search_seeds: list[int]) -> dict:
    reference = get_reference(reference_name)
    exact_summary = dict(invman_rust.perishable_inventory_exact_mdp_summary(reference_name))

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

    exact_return = float(exact_summary["value_iteration_mean_return"])
    best_heuristic_return = max(
        float(base_stock_eval["mean_return"]),
        float(bsp_low_ew_eval["mean_return"]),
    )

    learned_rows = []
    for leaf_type in parsed.leaf_types:
        learned_seed = stable_seed(parsed.seed, f"exact::{reference_name}::{leaf_type}")
        trained_model = train_soft_tree_on_reference(
            reference,
            depth=parsed.depth,
            temperature=parsed.temperature,
            split_type=parsed.split_type,
            leaf_type=leaf_type,
            training_episodes=parsed.exact_training_episodes,
            es_population=parsed.es_population,
            sigma_init=parsed.sigma_init,
            seed=learned_seed,
            same_seed=parsed.same_seed,
            run_tag=f"paper_exact_{reference_name}_d{parsed.depth}_{parsed.split_type}_{leaf_type}",
        )
        learned_eval = evaluate_soft_tree_policy(reference, trained_model, eval_seeds)
        learned_rows.append(
            {
                "policy": f"soft_tree_{leaf_type}",
                "params": f"d={parsed.depth}, leaf={leaf_type}",
                "mean_return": float(learned_eval["mean_return"]),
                "gap_to_exact_optimum": float(exact_return - float(learned_eval["mean_return"])),
                "gap_to_best_heuristic": float(best_heuristic_return - float(learned_eval["mean_return"])),
                "note": f"CMA-ES trained soft tree, seed={learned_seed}",
                "evaluation": learned_eval,
                "training": {
                    "episodes": parsed.exact_training_episodes,
                    "es_population": parsed.es_population,
                    "sigma_init": parsed.sigma_init,
                    "seed": learned_seed,
                    "same_seed": parsed.same_seed,
                },
            }
        )

    rows = [
        {
            "policy": "exact_value_iteration",
            "params": "-",
            "mean_return": exact_return,
            "gap_to_exact_optimum": 0.0,
            "gap_to_best_heuristic": float(best_heuristic_return - exact_return),
            "note": "Exact tabular MDP optimum",
        },
        {
            "policy": "base_stock",
            "params": str(base_stock_search["best"]["params"]),
            "mean_return": float(base_stock_eval["mean_return"]),
            "gap_to_exact_optimum": float(exact_return - float(base_stock_eval["mean_return"])),
            "gap_to_best_heuristic": float(best_heuristic_return - float(base_stock_eval["mean_return"])),
            "note": "Best heuristic on stochastic search seeds",
            "search": base_stock_search,
            "evaluation": base_stock_eval,
        },
        {
            "policy": "bsp_low_ew",
            "params": str(bsp_low_ew_search["best"]["params"]),
            "mean_return": float(bsp_low_ew_eval["mean_return"]),
            "gap_to_exact_optimum": float(exact_return - float(bsp_low_ew_eval["mean_return"])),
            "gap_to_best_heuristic": float(best_heuristic_return - float(bsp_low_ew_eval["mean_return"])),
            "note": "Best heuristic on stochastic search seeds",
            "search": bsp_low_ew_search,
            "evaluation": bsp_low_ew_eval,
        },
    ]
    rows.extend(learned_rows)

    return {
        "reference_instance_name": reference_name,
        "reference": reference,
        "exact_mdp": exact_summary,
        "best_heuristic_mean_return": best_heuristic_return,
        "rows": rows,
    }


def build_practical_slice(parsed) -> dict:
    dataset = load_dataset(parsed.practical_dataset)
    reference = get_reference(dataset["reference_instance_name"])
    if dataset["reference_instance_name"] != parsed.practical_reference:
        raise ValueError(
            "practical dataset reference instance "
            f"'{dataset['reference_instance_name']}' does not match --practical_reference '{parsed.practical_reference}'"
        )

    train_demands = [int(value) for value in dataset["train_demands"]]
    test_demands = [int(value) for value in dataset["test_demands"]]
    estimated_mean = empirical_mean(train_demands)

    base_stock_search = search_best_base_stock_from_demands(
        reference,
        train_demands,
        demand_mean=estimated_mean,
    )
    bsp_low_ew_search = search_best_bsp_low_ew_from_demands(
        reference,
        train_demands,
        demand_mean=estimated_mean,
    )

    policy_rows = [
        {
            "policy": "base_stock",
            "split": "train",
            "params": base_stock_search["best"]["params"],
            "metrics": evaluate_heuristic_trace_summary(
                reference,
                "base_stock",
                base_stock_search["best"]["params"],
                train_demands,
                demand_mean=estimated_mean,
            ),
            "notes": "train-trace tuned heuristic",
        },
        {
            "policy": "base_stock",
            "split": "test",
            "params": base_stock_search["best"]["params"],
            "metrics": evaluate_heuristic_trace_summary(
                reference,
                "base_stock",
                base_stock_search["best"]["params"],
                test_demands,
                demand_mean=estimated_mean,
            ),
            "notes": "held-out practical evaluation",
        },
        {
            "policy": "bsp_low_ew",
            "split": "train",
            "params": bsp_low_ew_search["best"]["params"],
            "metrics": evaluate_heuristic_trace_summary(
                reference,
                "bsp_low_ew",
                bsp_low_ew_search["best"]["params"],
                train_demands,
                demand_mean=estimated_mean,
            ),
            "notes": "train-trace tuned heuristic",
        },
        {
            "policy": "bsp_low_ew",
            "split": "test",
            "params": bsp_low_ew_search["best"]["params"],
            "metrics": evaluate_heuristic_trace_summary(
                reference,
                "bsp_low_ew",
                bsp_low_ew_search["best"]["params"],
                test_demands,
                demand_mean=estimated_mean,
            ),
            "notes": "held-out practical evaluation",
        },
    ]

    for leaf_type in parsed.leaf_types:
        learned_seed = stable_seed(parsed.seed, f"practical::{dataset['name']}::{leaf_type}")
        trained_model = train_soft_tree_on_demand_trace(
            reference,
            train_demands,
            demand_mean=estimated_mean,
            depth=parsed.depth,
            temperature=parsed.temperature,
            split_type=parsed.split_type,
            leaf_type=leaf_type,
            training_episodes=parsed.practical_training_episodes,
            es_population=parsed.es_population,
            sigma_init=parsed.sigma_init,
            seed=learned_seed,
            same_seed=parsed.same_seed,
            run_tag=f"paper_practical_{dataset['name']}_d{parsed.depth}_{parsed.split_type}_{leaf_type}",
        )
        train_metrics = evaluate_soft_tree_trace_summary(
            reference,
            trained_model,
            train_demands,
            demand_mean=estimated_mean,
        )
        test_metrics = evaluate_soft_tree_trace_summary(
            reference,
            trained_model,
            test_demands,
            demand_mean=estimated_mean,
        )
        policy_rows.extend(
            [
                {
                    "policy": f"soft_tree_{leaf_type}",
                    "split": "train",
                    "params": f"d={parsed.depth}, leaf={leaf_type}",
                    "metrics": train_metrics,
                    "notes": f"CMA-ES tuned on train trace, seed={learned_seed}",
                },
                {
                    "policy": f"soft_tree_{leaf_type}",
                    "split": "test",
                    "params": f"d={parsed.depth}, leaf={leaf_type}",
                    "metrics": test_metrics,
                    "notes": f"held-out practical evaluation, seed={learned_seed}",
                },
            ]
        )

    return {
        "dataset": dataset,
        "reference_instance_name": dataset["reference_instance_name"],
        "calibration_protocol": (
            "Tune heuristic parameters on the train demand trace with deterministic search. "
            "Train soft-tree policy parameters with CMA-ES directly on the train trace objective. "
            "Report both train and held-out test metrics."
        ),
        "dataset_diagnostics": {
            "train_mean_demand": estimated_mean,
            "test_mean_demand": empirical_mean(test_demands),
            "train_periods": len(train_demands),
            "test_periods": len(test_demands),
        },
        "metric_order": list(METRIC_ORDER),
        "metric_labels": dict(METRIC_LABELS),
        "policy_rows": policy_rows,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# perishable_inventory Paper Benchmark",
        "",
        "- objective: optimize structured policy classes with CMA-ES and compare them against heuristics and the exact optimum when available",
        f"- tree_depth: `{payload['tree_family']['depth']}`",
        f"- split_type: `{payload['tree_family']['split_type']}`",
        f"- leaf_types: `{payload['tree_family']['leaf_types']}`",
        "",
        "## Exact Slice",
        "",
    ]

    for instance in payload["exact_slice"]["instances"]:
        lines.extend(
            [
                f"### `{instance['reference_instance_name']}`",
                "",
                f"- exact_value_iteration_return: `{instance['exact_mdp']['value_iteration_mean_return']:.4f}`",
                f"- best_base_stock_level: `{instance['exact_mdp']['best_base_stock_level']}`",
                f"- best_heuristic_mean_return: `{instance['best_heuristic_mean_return']:.4f}`",
                "",
                "| Policy | Params | Mean Return | Gap to Exact Optimum | Gap to Best Heuristic | Note |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in instance["rows"]:
            lines.append(
                f"| `{row['policy']}` | `{row['params']}` | `{row['mean_return']:.4f}` | "
                f"`{row['gap_to_exact_optimum']:.4f}` | `{row['gap_to_best_heuristic']:.4f}` | {row['note']} |"
            )
        lines.append("")

    practical = payload["practical_slice"]
    dataset = practical["dataset"]
    lines.extend(
        [
            "## Practical Slice",
            "",
            f"- dataset: `{dataset['name']}`",
            f"- source_kind: `{dataset['source_kind']}`",
            f"- source_note: {dataset['source_note']}",
            f"- practical_goal: {dataset['practical_goal']}",
            f"- calibration: {practical['calibration_protocol']}",
        ]
    )
    for key, value in practical["dataset_diagnostics"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.4f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "| Policy | Split | Params | "
            + " | ".join(practical["metric_labels"][metric] for metric in practical["metric_order"])
            + " | Notes |",
            "| --- | --- | --- | "
            + " | ".join("---:" for _ in practical["metric_order"])
            + " | --- |",
        ]
    )
    for row in practical["policy_rows"]:
        rendered_metrics = []
        for metric in practical["metric_order"]:
            value = row["metrics"][metric]
            rendered_metrics.append(f"`{value:.4f}`" if isinstance(value, float) else f"`{value}`")
        lines.append(
            f"| `{row['policy']}` | `{row['split']}` | `{row['params']}` | "
            + " | ".join(rendered_metrics)
            + f" | {row['notes']} |"
        )
    return "\n".join(lines)


def write_report(payload: dict, *, output_json: str | Path, output_markdown: str | Path) -> dict:
    markdown = render_markdown(payload)
    payload = dict(payload)
    payload["markdown"] = markdown
    output_json_path = Path(output_json)
    output_markdown_path = Path(output_markdown)
    ensure_parent(output_json_path)
    ensure_parent(output_markdown_path)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return payload


def main():
    parsed = parse_args()
    eval_seeds = [parsed.seed + offset for offset in range(parsed.eval_seeds)]
    heuristic_search_seeds = [parsed.seed + offset for offset in range(parsed.heuristic_search_seeds)]

    exact_instances = [
        build_exact_slice_rows(reference_name, parsed, eval_seeds, heuristic_search_seeds)
        for reference_name in parsed.exact_references
    ]
    practical_slice = build_practical_slice(parsed)

    payload = {
        "family": "perishable_inventory",
        "benchmark": "paper_benchmark",
        "paper_objective": (
            "Design structured policy classes for perishable inventory, optimize their parameters "
            "with CMA-ES, and compare them against benchmark heuristics plus the exact optimum on "
            "small instances."
        ),
        "tree_family": {
            "policy_family": "soft_tree",
            "depth": parsed.depth,
            "temperature": parsed.temperature,
            "split_type": parsed.split_type,
            "leaf_types": list(parsed.leaf_types),
            "exact_training_episodes": parsed.exact_training_episodes,
            "practical_training_episodes": parsed.practical_training_episodes,
            "es_population": parsed.es_population,
            "sigma_init": parsed.sigma_init,
            "seed": parsed.seed,
            "same_seed": parsed.same_seed,
        },
        "exact_slice": {
            "reported_instances": list(parsed.exact_references),
            "eval_seeds": eval_seeds,
            "heuristic_search_seeds": heuristic_search_seeds,
            "instances": exact_instances,
        },
        "practical_slice": practical_slice,
    }

    payload = write_report(
        payload,
        output_json=parsed.output_json,
        output_markdown=parsed.output_markdown,
    )
    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
