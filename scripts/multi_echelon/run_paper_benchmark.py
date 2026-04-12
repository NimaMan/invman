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

from invman.es_mp import train

from common import (
    action_grids,
    benchmark_allocation_mode,
    benchmark_periods,
    benchmark_replications,
    build_soft_tree_model,
    confidence_half_width,
    dumps_json,
    ensure_parent,
    evaluate_soft_tree_policy,
    evaluate_stationary_policy,
    get_benchmark_reference,
    get_reference,
    initialize_soft_tree_to_constant_action,
    list_references,
    savings_pct,
    savings_pct_samples,
    search_best_constant_base_stock,
    soft_tree_rollout_kwargs,
)

import invman_rust


DEFAULT_OUTPUT_JSON = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "multi_echelon"
    / "divergent_special_delivery"
    / "experiments"
    / "reports"
    / "latest_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "multi_echelon"
    / "divergent_special_delivery"
    / "experiments"
    / "reports"
    / "README.md"
)

DEFAULT_ARTIFACT_DIR = PACKAGE_ROOT / "outputs" / "multi_echelon" / "paper_benchmark"
BASELINE_NOTE = (
    "Repo comparator is the best constant base-stock policy searched over the carried Van Roy action grid. "
    "The Gijs text clearly states the learned policy uses that grid, but the constant-base-stock search domain "
    "in the paper still needs final clarification."
)
ALGORITHM_VERIFICATION_NOTE = (
    "The Gijs paper reports only relative savings for the two carried settings, not absolute constant base-stock means. "
    "Absolute heuristic and NDP verification for this family comes from the original Van Roy case-study rows."
)


def stable_seed(base_seed: int, tag: str) -> int:
    return int(base_seed + (zlib.adler32(tag.encode("utf-8")) % 1_000_000))


def repo_algorithm_verification_rows() -> list[dict]:
    return [
        {
            "algorithm": "constant_base_stock",
            "row_source": "repo_algorithm_result",
            "literature_verified": False,
            "verification_anchor": None,
            "note": ALGORITHM_VERIFICATION_NOTE,
        }
    ]


def published_policy_rows(reference: dict) -> list[dict]:
    return [
        {
            "policy": "published_a3c",
            "row_source": "published_literature_row",
            "reported_savings_pct": float(reference["published_a3c_savings_pct"]),
            "reported_half_width_pct": float(reference["published_a3c_confidence_half_width_pct"]),
            "note": "Published Gijs benchmark row.",
        },
        {
            "policy": "published_van_roy_ndp",
            "row_source": "published_literature_row",
            "reported_savings_pct": float(reference["published_van_roy_savings_pct_approx"]),
            "reported_half_width_pct": None,
            "note": "Published Van Roy benchmark row, reported approximately.",
        },
    ]


def default_instance_names() -> list[str]:
    return [str(reference["name"]) for reference in list_references()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Gijs / Van Roy multi-echelon paper benchmark with literature-matched evaluation and CMA-ES soft-tree policies."
    )
    parser.add_argument("--instance_names", nargs="+", default=default_instance_names())
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
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
        choices=["direct_base_stock", "anchor_adjustment"],
        default="anchor_adjustment",
    )
    parser.add_argument("--training_horizon", type=int, default=10_000)
    parser.add_argument("--training_episodes", type=int, default=400)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_seed_batch", type=int, default=8)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--heuristic_search_replications", type=int, default=None)
    parser.add_argument("--heuristic_finalist_top_k", type=int, default=10)
    parser.add_argument("--benchmark_replications", type=int, default=None)
    parser.add_argument("--artifact_dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser.parse_args()


def training_namespace(parsed, reference: dict, *, run_tag: str, seed: int) -> SimpleNamespace:
    output_root = Path(parsed.artifact_dir) / reference["name"] / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(parsed.training_horizon),
        seed=int(seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def get_model_fitness(model, reference: dict, training_horizon: int):
    rollout_overrides = dict(getattr(model, "_multi_echelon_rollout_overrides", {}))

    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        rollout_kwargs = soft_tree_rollout_kwargs(
            reference,
            model,
            flat_params=flat_params,
            include_period_feature=False,
            **rollout_overrides,
        )
        rollout_kwargs["horizon"] = int(training_horizon)
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            cost = invman_rust.multi_echelon_soft_tree_rollout(
                seed=int(seed) + seed_offset,
                **rollout_kwargs,
            )
            costs.append(float(cost))
        mean_cost = float(np.mean(costs))
        reward = -mean_cost
        if verbose:
            print(f"Seed {seed}: cost {mean_cost:.4f}, reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def get_population_fitness(model, reference: dict, training_horizon: int):
    rollout_overrides = dict(getattr(model, "_multi_echelon_rollout_overrides", {}))

    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(params, dtype=np.float32).tolist() for params in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=model.get_model_flat_params(),
                include_period_feature=False,
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


def instance_result(parsed, reference_name: str) -> dict:
    reference = get_reference(reference_name)
    benchmark_reps = int(parsed.benchmark_replications or benchmark_replications(reference))
    heuristic_search_reps = int(parsed.heuristic_search_replications or min(benchmark_reps, 16))
    heuristic_seed = stable_seed(int(parsed.seed), f"{reference_name}:heuristic")
    train_seed = stable_seed(int(parsed.seed), f"{reference_name}:train")
    eval_seed_start = stable_seed(1_000_000, f"{reference_name}:eval")
    eval_seeds = [eval_seed_start + idx for idx in range(benchmark_reps)]

    heuristic_search = search_best_constant_base_stock(
        reference,
        allocation_mode=benchmark_allocation_mode(reference),
        replications=heuristic_search_reps,
        seed=heuristic_seed,
        top_k=10,
    )
    heuristic_candidates = [dict(row) for row in heuristic_search["top_results"][: int(parsed.heuristic_finalist_top_k)]]
    heuristic_candidate_evals = []
    for candidate in heuristic_candidates:
        candidate_costs = []
        for seed in eval_seeds:
            candidate_costs.append(
                float(
                    evaluate_stationary_policy(
                        reference,
                        warehouse_level=int(candidate["warehouse_level"]),
                        retailer_level=int(candidate["retailer_level"]),
                        allocation_mode=benchmark_allocation_mode(reference),
                        replications=1,
                        seed=int(seed),
                    )["mean_cost"]
                )
            )
        heuristic_candidate_evals.append(
            {
                "warehouse_level": int(candidate["warehouse_level"]),
                "retailer_level": int(candidate["retailer_level"]),
                "mean_cost": float(np.mean(candidate_costs)),
                "cost_std": float(np.std(candidate_costs)),
                "num_samples": int(len(candidate_costs)),
                "costs": candidate_costs,
            }
        )
    heuristic_best = min(heuristic_candidate_evals, key=lambda row: float(row["mean_cost"]))
    benchmark_warehouse_levels, benchmark_retailer_levels = action_grids(reference)
    warehouse_anchor_level = int(heuristic_best["warehouse_level"])
    retailer_anchor_level = int(heuristic_best["retailer_level"])
    warehouse_adjustments = sorted(
        {
            int(warehouse_anchor_level - level)
            for level in benchmark_warehouse_levels
            if int(level) <= warehouse_anchor_level
        }
    )
    retailer_adjustments = sorted(
        {
            int(level - retailer_anchor_level)
            for level in benchmark_retailer_levels
            if int(level) >= retailer_anchor_level
        }
    )
    policy_feature_mode = str(parsed.policy_feature_mode)
    policy_action_mode = str(parsed.policy_action_mode)
    rollout_overrides = {
        "policy_feature_mode": policy_feature_mode,
        "policy_action_mode": policy_action_mode,
        "warehouse_anchor_level": warehouse_anchor_level,
        "retailer_anchor_level": retailer_anchor_level,
        "reference_warehouse_levels": benchmark_warehouse_levels,
        "reference_retailer_levels": benchmark_retailer_levels,
    }

    model_kwargs = {
        "policy_feature_mode": policy_feature_mode,
    }
    init_action = [warehouse_anchor_level, retailer_anchor_level]
    if policy_action_mode == "anchor_adjustment":
        model_kwargs["warehouse_levels"] = warehouse_adjustments
        model_kwargs["retailer_levels"] = retailer_adjustments
        init_action = [0, 0]

    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
        include_period_feature=False,
        **model_kwargs,
    )
    initialize_soft_tree_to_constant_action(model, init_action)
    model._multi_echelon_rollout_overrides = dict(rollout_overrides)
    run_tag = (
        f"{reference_name}_{policy_feature_mode}_{policy_action_mode}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_t{int(round(parsed.temperature * 1000)):03d}_s{parsed.seed}_b{parsed.train_seed_batch}"
    )
    train_args = training_namespace(parsed, reference, run_tag=run_tag, seed=train_seed)
    training_horizon = int(parsed.training_horizon)
    trained_model, _ = train(
        model=model,
        get_model_fitness=get_model_fitness(model, reference, training_horizon),
        get_population_fitness=get_population_fitness(model, reference, training_horizon),
        args=train_args,
        same_seed=bool(parsed.same_seed),
    )

    learned_eval = evaluate_soft_tree_policy(
        reference,
        trained_model,
        eval_seeds,
        include_period_feature=False,
        **rollout_overrides,
    )

    heuristic_eval_costs = np.asarray(heuristic_best["costs"], dtype=np.float64)
    learned_costs = np.asarray(learned_eval["costs"], dtype=np.float64)
    learned_savings = savings_pct(float(np.mean(heuristic_eval_costs)), float(np.mean(learned_costs)))
    learned_savings_samples = savings_pct_samples(heuristic_eval_costs.tolist(), learned_costs.tolist())

    return {
        "reference": reference,
        "repo_algorithm_rows": repo_algorithm_verification_rows(),
        "published_policy_rows": published_policy_rows(reference),
        "heuristic_search": heuristic_search,
        "heuristic_eval": {
            "mean_cost": float(np.mean(heuristic_eval_costs)),
            "cost_std": float(np.std(heuristic_eval_costs)),
            "num_samples": int(heuristic_eval_costs.size),
            "costs": heuristic_eval_costs.tolist(),
        },
        "heuristic_finalist_evaluations": heuristic_candidate_evals,
        "heuristic_best": {
            "warehouse_level": int(heuristic_best["warehouse_level"]),
            "retailer_level": int(heuristic_best["retailer_level"]),
            "mean_cost": float(heuristic_best["mean_cost"]),
            "cost_std": float(heuristic_best["cost_std"]),
            "num_samples": int(heuristic_best["num_samples"]),
        },
        "soft_tree_eval": learned_eval,
        "soft_tree_savings_pct": learned_savings,
        "soft_tree_savings_confidence_half_width_pct": confidence_half_width(learned_savings_samples),
        "published_a3c_savings_pct": float(reference["published_a3c_savings_pct"]),
        "published_a3c_confidence_half_width_pct": float(reference["published_a3c_confidence_half_width_pct"]),
        "published_van_roy_savings_pct_approx": float(reference["published_van_roy_savings_pct_approx"]),
        "gaps": {
            "vs_published_a3c_pct": float(learned_savings - reference["published_a3c_savings_pct"]),
            "vs_published_van_roy_pct": float(learned_savings - reference["published_van_roy_savings_pct_approx"]),
        },
        "policy_config": {
            "depth": int(parsed.depth),
            "temperature": float(parsed.temperature),
            "split_type": str(parsed.split_type),
            "leaf_type": str(parsed.leaf_type),
            "policy_feature_mode": policy_feature_mode,
            "policy_action_mode": policy_action_mode,
        },
        "initialization": {
            "kind": "best_constant_base_stock",
            "levels": [int(heuristic_best["warehouse_level"]), int(heuristic_best["retailer_level"])],
            "adjustment_levels": {
                "warehouse": warehouse_adjustments,
                "retailer": retailer_adjustments,
            },
        },
        "training_budget": {
            "training_horizon": int(parsed.training_horizon),
            "training_episodes": int(parsed.training_episodes),
            "es_population": int(parsed.es_population),
            "sigma_init": float(parsed.sigma_init),
            "train_seed_batch": int(parsed.train_seed_batch),
            "heuristic_search_replications": int(heuristic_search_reps),
            "heuristic_finalist_top_k": int(parsed.heuristic_finalist_top_k),
        },
    }


def instance_table(instance_results: list[dict]) -> list[dict]:
    rows = []
    for result in instance_results:
        reference = result["reference"]
        rows.append(
            {
                "instance_name": str(reference["name"]),
                "base_stock_cost": float(result["heuristic_eval"]["mean_cost"]),
                "soft_tree_cost": float(result["soft_tree_eval"]["mean_cost"]),
                "soft_tree_savings_pct": float(result["soft_tree_savings_pct"]),
                "soft_tree_savings_half_width_pct": float(result["soft_tree_savings_confidence_half_width_pct"]),
                "repo_algorithm_rows": result["repo_algorithm_rows"],
                "published_policy_rows": result["published_policy_rows"],
                "published_a3c_savings_pct": float(result["published_a3c_savings_pct"]),
                "published_a3c_half_width_pct": float(result["published_a3c_confidence_half_width_pct"]),
                "published_van_roy_savings_pct_approx": float(result["published_van_roy_savings_pct_approx"]),
                "gap_vs_published_a3c_pct": float(result["gaps"]["vs_published_a3c_pct"]),
            }
        )
    return rows


def aggregate_summary(rows: list[dict]) -> dict:
    return {
        "num_instances": int(len(rows)),
        "beats_published_a3c_count": int(
            sum(row["soft_tree_savings_pct"] > row["published_a3c_savings_pct"] for row in rows)
        ),
        "beats_published_van_roy_count": int(
            sum(row["soft_tree_savings_pct"] > row["published_van_roy_savings_pct_approx"] for row in rows)
        ),
        "mean_soft_tree_savings_pct": float(np.mean([row["soft_tree_savings_pct"] for row in rows])),
        "mean_gap_vs_published_a3c_pct": float(np.mean([row["gap_vs_published_a3c_pct"] for row in rows])),
    }


def report_markdown(payload: dict) -> str:
    lines = [
        "# multi_echelon Paper Benchmark Report",
        "",
        f"- source: {payload['source']}",
        f"- url: {payload['url']}",
        f"- instances: `{len(payload['instance_results'])}`",
        f"- policy family: depth `{payload['policy_config']['depth']}` `{payload['policy_config']['split_type']}` soft tree with `{payload['policy_config']['leaf_type']}` leaves",
        f"- literature evaluation: `{payload['protocol']['benchmark_replications']}` sample paths of `{payload['protocol']['benchmark_periods']}` periods each",
        f"- literature baseline: constant base-stock with `{payload['protocol']['allocation_mode']}` allocation",
        f"- baseline note: {payload['protocol']['baseline_note']}",
        f"- training budget: `{payload['protocol']['training_episodes']}` CMA-ES episodes of length `{payload['protocol']['training_horizon']}`",
        "",
        "## Reporting Rule",
        "",
        "- `literature_verified` applies only to repo exact or heuristic algorithms.",
        "- Published A3C / PPO / NDP rows from papers are carried as published rows, not as verified repo algorithms.",
        "- Repo reproduced absolute costs are shown separately from published literature numbers.",
        "",
        "## Aggregate",
        "",
        f"- beats published A3C savings on `{payload['aggregate']['beats_published_a3c_count']}` / `{payload['aggregate']['num_instances']}` settings",
        f"- beats published Van Roy savings on `{payload['aggregate']['beats_published_van_roy_count']}` / `{payload['aggregate']['num_instances']}` settings",
        f"- mean soft-tree savings vs repo constant base-stock: `{payload['aggregate']['mean_soft_tree_savings_pct']:.3f}%`",
        f"- mean gap vs published A3C savings: `{payload['aggregate']['mean_gap_vs_published_a3c_pct']:.3f}` percentage points",
        "",
        "## Repo Algorithm Verification",
        "",
        "| Repo Algorithm | literature_verified | Verification Anchor | Note |",
        "| --- | --- | --- | --- |",
        f"| `constant_base_stock` | `False` | `none` | {ALGORITHM_VERIFICATION_NOTE} |",
        "",
        "## Published Numbers Confirmed",
        "",
        "| Instance | Published Constant Base-Stock Cost | Published A3C Savings | Published Van Roy Savings |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["instance_table"]:
        lines.append(
            f"| `{row['instance_name']}` | `not reported` | "
            f"`{row['published_a3c_savings_pct']:.2f}% +/- {row['published_a3c_half_width_pct']:.2f}%` | "
            f"`~{row['published_van_roy_savings_pct_approx']:.2f}%` |"
        )
    lines += [
        "",
        "## Per Instance",
        "",
        "Repo reproduction benchmark:",
        "",
        "| Instance | Base-Stock Cost | Soft Tree Cost | Soft Tree Savings | 95% Half-Width | Published A3C Savings | Published Van Roy Savings | Gap vs A3C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["instance_table"]:
        lines.append(
            f"| `{row['instance_name']}` | `{row['base_stock_cost']:.3f}` | `{row['soft_tree_cost']:.3f}` | "
            f"`{row['soft_tree_savings_pct']:.3f}%` | `{row['soft_tree_savings_half_width_pct']:.3f}%` | "
            f"`{row['published_a3c_savings_pct']:.2f}% +/- {row['published_a3c_half_width_pct']:.2f}%` | "
            f"`~{row['published_van_roy_savings_pct_approx']:.2f}%` | `{row['gap_vs_published_a3c_pct']:.3f}` |"
        )
    return "\n".join(lines)


def write_progress(payload: dict, *, output_json: str | Path, output_markdown: str | Path):
    resolved_json = Path(output_json)
    ensure_parent(resolved_json)
    resolved_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    resolved_markdown = Path(output_markdown)
    ensure_parent(resolved_markdown)
    resolved_markdown.write_text(report_markdown(payload) + "\n", encoding="utf-8")


def main():
    parsed = parse_args()
    benchmark_reference = get_benchmark_reference()
    instance_results = []
    for reference_name in parsed.instance_names:
        result = instance_result(parsed, reference_name)
        instance_results.append(result)
        partial_payload = {
            "source": benchmark_reference["source"],
            "url": benchmark_reference["url"],
            "policy_config": {
                "depth": int(parsed.depth),
                "temperature": float(parsed.temperature),
                "split_type": str(parsed.split_type),
                "leaf_type": str(parsed.leaf_type),
            },
            "protocol": {
                "benchmark_periods": int(benchmark_periods(get_reference(reference_name))),
                "benchmark_replications": int(parsed.benchmark_replications or benchmark_replications(get_reference(reference_name))),
                "allocation_mode": benchmark_allocation_mode(get_reference(reference_name)),
                "baseline_note": BASELINE_NOTE,
                "training_horizon": int(parsed.training_horizon),
                "training_episodes": int(parsed.training_episodes),
            },
            "instance_results": instance_results,
            "instance_table": instance_table(instance_results),
            "aggregate": aggregate_summary(instance_table(instance_results)),
        }
        write_progress(partial_payload, output_json=parsed.output_json, output_markdown=parsed.output_markdown)

    payload = {
        "source": benchmark_reference["source"],
        "url": benchmark_reference["url"],
        "policy_config": {
            "depth": int(parsed.depth),
            "temperature": float(parsed.temperature),
            "split_type": str(parsed.split_type),
            "leaf_type": str(parsed.leaf_type),
        },
        "protocol": {
            "benchmark_periods": int(benchmark_periods(get_reference(parsed.instance_names[0]))),
            "benchmark_replications": int(parsed.benchmark_replications or benchmark_replications(get_reference(parsed.instance_names[0]))),
            "allocation_mode": benchmark_allocation_mode(get_reference(parsed.instance_names[0])),
            "baseline_note": BASELINE_NOTE,
            "training_horizon": int(parsed.training_horizon),
            "training_episodes": int(parsed.training_episodes),
        },
        "instance_results": instance_results,
        "instance_table": instance_table(instance_results),
        "aggregate": aggregate_summary(instance_table(instance_results)),
    }
    write_progress(payload, output_json=parsed.output_json, output_markdown=parsed.output_markdown)
    print(dumps_json(payload))
    print()
    print(report_markdown(payload))


if __name__ == "__main__":
    main()
