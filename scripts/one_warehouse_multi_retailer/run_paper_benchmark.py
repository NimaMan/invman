from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cpu_limits import configure_process_cpu_limits_from_argv, normalize_args_cpu_limits

configure_process_cpu_limits_from_argv(sys.argv[1:])

import numpy as np

from invman.es_mp import train

from common import (
    benchmark_initial_state,
    benchmark_periods,
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_echelon_base_stock_policy,
    evaluate_soft_tree_policy,
    get_reference,
    list_references,
    policy_action_mode_for_reference,
    published_cost,
    search_best_echelon_base_stock,
    soft_tree_rollout_kwargs,
    uses_kaynov_k_search,
)

import invman_rust


DEFAULT_OUTPUT_JSON = (
    PACKAGE_ROOT
    / "src"
    / "problems"
    / "one_warehouse_multi_retailer"
    / "experiments"
    / "reports"
    / "latest_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = (
    PACKAGE_ROOT
    / "src"
    / "problems"
    / "one_warehouse_multi_retailer"
    / "experiments"
    / "reports"
    / "README.md"
)

DEFAULT_ARTIFACT_DIR = (
    PACKAGE_ROOT / "outputs" / "one_warehouse_multi_retailer" / "paper_benchmark"
)


def stable_seed(base_seed: int, tag: str) -> int:
    return int(base_seed + (zlib.adler32(tag.encode("utf-8")) % 1_000_000))


def default_instance_names() -> list[str]:
    references = list_references()
    return [str(reference["name"]) for reference in references]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Kaynov OWMR paper benchmark with the literature evaluation protocol and CMA-ES soft-tree policies."
    )
    parser.add_argument("--instance_names", nargs="+", default=default_instance_names())
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--training_episodes_small", type=int, default=300)
    parser.add_argument("--training_episodes_large", type=int, default=400)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_seed_batch", type=int, default=16)
    parser.add_argument("--train_allocation_policy", default="random_sequential")
    parser.add_argument("--eval_allocation_policy", default="proportional")
    parser.add_argument("--heuristic_search_replications", type=int, default=1000)
    parser.add_argument("--benchmark_replications", type=int, default=1000)
    parser.add_argument("--eval_seeds", type=int, default=1000)
    parser.add_argument("--eval_seed_start", type=int, default=1_000_000)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--artifact_dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser.parse_args()


def _training_episodes(reference: dict, parsed) -> int:
    return (
        int(parsed.training_episodes_large)
        if len(reference["retailer_lead_times"]) >= 10
        else int(parsed.training_episodes_small)
    )


def _training_namespace(
    parsed,
    reference: dict,
    *,
    run_tag: str,
    training_episodes: int,
    seed: int,
):
    output_root = Path(parsed.artifact_dir) / reference["name"] / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(training_episodes),
        mp_num_processors=1,
        save_every=max(1, int(training_episodes)),
        save_solutions=False,
        horizon=int(benchmark_periods(reference)),
        seed=int(seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_model_fitness(model, reference: dict, allocation_policy: str, policy_action_mode: str):
    def inner(
        _model,
        args,
        model_params=None,
        seed=1234,
        indiv_idx=-1,
        return_env=False,
        track_demand=False,
        verbose=False,
    ):
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
        mean_cost = float(np.mean(costs))
        reward = -mean_cost
        if verbose:
            print(f"Seed {seed}: cost {mean_cost:.4f}, reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def _get_population_fitness(
    model,
    reference: dict,
    allocation_policy: str,
    policy_action_mode: str,
):
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
        return [(-float(discounted_cost), indiv_idx) for indiv_idx, discounted_cost in enumerate(costs.tolist())]

    return inner


def _comparison_rows(
    reference: dict,
    repo_heuristics: dict,
    learned_evaluation: dict,
) -> list[dict]:
    rows = [
        {
            "policy": "published_echelon_base_stock_proportional",
            "mean_cost": float(published_cost(reference["published_proportional_benchmark"])),
            "note": "Kaynov Table A.3",
        },
        {
            "policy": "published_echelon_base_stock_min_shortage",
            "mean_cost": float(published_cost(reference["published_min_shortage_benchmark"])),
            "note": "Kaynov Table A.3",
        },
        {
            "policy": "published_ppo",
            "mean_cost": float(published_cost(reference["published_ppo_benchmark"])),
            "note": "Kaynov Table A.3",
        },
        {
            "policy": "repo_echelon_base_stock_proportional",
            "mean_cost": float(repo_heuristics["proportional"]["mean_cost"]),
            "note": "repo reproduction",
        },
        {
            "policy": "repo_echelon_base_stock_min_shortage",
            "mean_cost": float(repo_heuristics["min_shortage"]["mean_cost"]),
            "note": "repo reproduction",
        },
        {
            "policy": "soft_tree",
            "mean_cost": float(learned_evaluation["mean_cost"]),
            "note": "CMA-ES soft tree",
        },
    ]
    learned_cost = float(learned_evaluation["mean_cost"])
    published_best = min(
        float(published_cost(reference["published_proportional_benchmark"])),
        float(published_cost(reference["published_min_shortage_benchmark"])),
    )
    repo_best = min(
        float(repo_heuristics["proportional"]["mean_cost"]),
        float(repo_heuristics["min_shortage"]["mean_cost"]),
    )
    for row in rows:
        row["gap_vs_soft_tree_cost"] = float(row["mean_cost"] - learned_cost)
    rows.append(
        {
            "policy": "summary_best_published_heuristic",
            "mean_cost": published_best,
            "note": "min(published proportional, published min-shortage)",
            "gap_vs_soft_tree_cost": float(published_best - learned_cost),
        }
    )
    rows.append(
        {
            "policy": "summary_best_repo_heuristic",
            "mean_cost": repo_best,
            "note": "min(repo proportional, repo min-shortage)",
            "gap_vs_soft_tree_cost": float(repo_best - learned_cost),
        }
    )
    return rows


def _instance_markdown(instance_rows: list[dict]) -> str:
    lines = [
        "| Instance | CB | Learned | Best Repo Heuristic | Best Published Heuristic | Published PPO | Gap vs Repo Best | Gap vs PPO |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in instance_rows:
        lines.append(
            f"| `{row['instance_name']}` | `{row['customer_behavior']}` | "
            f"`{row['soft_tree_mean_cost']:.3f}` | "
            f"`{row['best_repo_heuristic_cost']:.3f}` | "
            f"`{row['best_published_heuristic_cost']:.3f}` | "
            f"`{row['published_ppo_cost']:.3f}` | "
            f"`{row['gap_vs_repo_best']:.3f}` | "
            f"`{row['gap_vs_published_ppo']:.3f}` |"
        )
    return "\n".join(lines)


def _report_markdown(payload: dict) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# one_warehouse_multi_retailer Paper Benchmark",
        "",
        f"- source: {payload['source']}",
        f"- url: {payload['url']}",
        f"- instances: `{len(payload['instance_results'])}`",
        f"- policy family: depth `{payload['policy_config']['depth']}` `{payload['policy_config']['split_type']}` soft tree with `{payload['policy_config']['leaf_type']}` leaves",
        f"- training allocation: `{payload['protocol']['train_allocation_policy']}`",
        f"- evaluation allocation: `{payload['protocol']['eval_allocation_policy']}`",
        f"- heuristic search: `{payload['protocol']['heuristic_search_replications']}` trajectories of length `{payload['protocol']['benchmark_periods']}` with common random numbers",
        f"- benchmark evaluation: `{payload['protocol']['benchmark_replications']}` independent trajectories of length `{payload['protocol']['benchmark_periods']}`",
        f"- instance 14 search note: {payload['protocol']['instance14_search_note']}",
        "",
        "## Aggregate",
        "",
        f"- beats best repo heuristic on `{aggregate['beats_best_repo_heuristic_count']}` / `{aggregate['num_instances']}` instances",
        f"- beats best published heuristic on `{aggregate['beats_best_published_heuristic_count']}` / `{aggregate['num_instances']}` instances",
        f"- beats published PPO on `{aggregate['beats_published_ppo_count']}` / `{aggregate['num_instances']}` instances",
        f"- mean gap vs best repo heuristic: `{aggregate['mean_gap_vs_repo_best']:.3f}`",
        f"- mean gap vs published PPO: `{aggregate['mean_gap_vs_published_ppo']:.3f}`",
        "",
        "## Per Instance",
        "",
        _instance_markdown(payload["instance_table"]),
    ]
    return "\n".join(lines)


def _write_progress(payload: dict, *, output_json: str | Path, output_markdown: str | Path):
    resolved_json = Path(output_json)
    ensure_parent(resolved_json)
    resolved_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    resolved_markdown = Path(output_markdown)
    ensure_parent(resolved_markdown)
    resolved_markdown.write_text(_report_markdown(payload) + "\n", encoding="utf-8")


def _instance_result(parsed, reference_name: str) -> dict:
    reference = get_reference(reference_name)
    policy_action_mode = policy_action_mode_for_reference(reference)
    train_seed = stable_seed(int(parsed.seed), f"{reference_name}:train")
    search_seed = stable_seed(int(parsed.seed), f"{reference_name}:heuristic_search")
    benchmark_seed = stable_seed(int(parsed.seed), f"{reference_name}:benchmark_eval")
    eval_seed_start = int(parsed.eval_seed_start) + stable_seed(0, reference_name)

    heuristic_search_proportional = search_best_echelon_base_stock(
        reference,
        allocation_policy="proportional",
        replications=int(parsed.heuristic_search_replications),
        seed=search_seed,
    )
    heuristic_search_min_shortage = search_best_echelon_base_stock(
        reference,
        allocation_policy="min_shortage",
        replications=int(parsed.heuristic_search_replications),
        seed=search_seed,
    )

    heuristics = {
        "proportional": evaluate_echelon_base_stock_policy(
            reference,
            warehouse_base_stock_level=heuristic_search_proportional["warehouse_base_stock_level"],
            retailer_base_stock_levels=heuristic_search_proportional["retailer_base_stock_levels"],
            allocation_policy="proportional",
            replications=int(parsed.benchmark_replications),
            seed=benchmark_seed,
        ),
        "min_shortage": evaluate_echelon_base_stock_policy(
            reference,
            warehouse_base_stock_level=heuristic_search_min_shortage["warehouse_base_stock_level"],
            retailer_base_stock_levels=heuristic_search_min_shortage["retailer_base_stock_levels"],
            allocation_policy="min_shortage",
            replications=int(parsed.benchmark_replications),
            seed=benchmark_seed,
        ),
    }

    model = build_soft_tree_model(
        reference,
        depth=int(parsed.depth),
        temperature=float(parsed.temperature),
        split_type=str(parsed.split_type),
        leaf_type=str(parsed.leaf_type),
        policy_action_mode=policy_action_mode,
    )
    run_tag = (
        f"{reference_name}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_{policy_action_mode}_train_{parsed.train_allocation_policy}_eval_{parsed.eval_allocation_policy}"
    )
    training_args = _training_namespace(
        parsed,
        reference,
        run_tag=run_tag,
        training_episodes=_training_episodes(reference, parsed),
        seed=train_seed,
    )
    trained_model, _ = train(
        model=model,
        get_model_fitness=_get_model_fitness(
            model,
            reference,
            str(parsed.train_allocation_policy),
            policy_action_mode,
        ),
        get_population_fitness=_get_population_fitness(
            model,
            reference,
            str(parsed.train_allocation_policy),
            policy_action_mode,
        ),
        args=training_args,
        same_seed=True,
    )
    eval_seeds = [eval_seed_start + offset for offset in range(int(parsed.eval_seeds))]
    learned_evaluation = evaluate_soft_tree_policy(
        reference,
        trained_model,
        eval_seeds,
        allocation_policy=str(parsed.eval_allocation_policy),
        policy_action_mode=policy_action_mode,
    )

    comparison_rows = _comparison_rows(reference, heuristics, learned_evaluation)
    published_ppo_cost = float(published_cost(reference["published_ppo_benchmark"]))
    best_published_heuristic_cost = min(
        float(published_cost(reference["published_proportional_benchmark"])),
        float(published_cost(reference["published_min_shortage_benchmark"])),
    )
    best_repo_heuristic_cost = min(
        float(heuristics["proportional"]["mean_cost"]),
        float(heuristics["min_shortage"]["mean_cost"]),
    )

    return {
        "reference": reference,
        "initial_state": benchmark_initial_state(reference),
        "search_results": {
            "proportional": heuristic_search_proportional,
            "min_shortage": heuristic_search_min_shortage,
        },
        "heuristics": heuristics,
        "policy_config": {
            "depth": int(parsed.depth),
            "temperature": float(parsed.temperature),
            "split_type": str(parsed.split_type),
            "leaf_type": str(parsed.leaf_type),
            "policy_action_mode": policy_action_mode,
            "training_episodes": int(_training_episodes(reference, parsed)),
            "es_population": int(parsed.es_population),
            "sigma_init": float(parsed.sigma_init),
            "train_seed_batch": int(parsed.train_seed_batch),
        },
        "training_seed": train_seed,
        "heuristic_search_seed": search_seed,
        "benchmark_seed": benchmark_seed,
        "eval_seed_start": eval_seed_start,
        "evaluation": learned_evaluation,
        "comparison_rows": comparison_rows,
        "summary": {
            "instance_name": reference_name,
            "customer_behavior": str(reference["customer_behavior"]),
            "soft_tree_mean_cost": float(learned_evaluation["mean_cost"]),
            "best_repo_heuristic_cost": best_repo_heuristic_cost,
            "best_published_heuristic_cost": best_published_heuristic_cost,
            "published_ppo_cost": published_ppo_cost,
            "gap_vs_repo_best": float(best_repo_heuristic_cost - learned_evaluation["mean_cost"]),
            "gap_vs_published_best_heuristic": float(best_published_heuristic_cost - learned_evaluation["mean_cost"]),
            "gap_vs_published_ppo": float(published_ppo_cost - learned_evaluation["mean_cost"]),
        },
    }


def main():
    parsed = parse_args()
    normalize_args_cpu_limits(parsed)
    references = [get_reference(name) for name in parsed.instance_names]
    payload = {
        "family": "one_warehouse_multi_retailer",
        "source": references[0]["source"] if references else "",
        "url": references[0]["url"] if references else "",
        "protocol": {
            "paper_reference": "Kaynov et al. (2024), Section 4.1, Table A.3, Table B.6",
            "benchmark_periods": 100,
            "heuristic_search_replications": int(parsed.heuristic_search_replications),
            "benchmark_replications": int(parsed.benchmark_replications),
            "eval_seeds": int(parsed.eval_seeds),
            "train_allocation_policy": str(parsed.train_allocation_policy),
            "eval_allocation_policy": str(parsed.eval_allocation_policy),
            "mp_num_processors": int(parsed.mp_num_processors),
            "thread_env": {
                "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
            "paper_evaluation_note": (
                "Paper heuristics use 1000 trajectories of length 100 with common random numbers during search; "
                "PPO and benchmark policies are evaluated on 1000 trajectories of length 100 using proportional allocation."
            ),
            "instance14_search_note": (
                "Kaynov state that instance 14 searches over warehouse level z0 and a shared percentile parameter k. "
                "The paper does not publish a discrete k-grid, so the repo enumerates the unique integer retailer-target vectors induced by continuous k in [0, 3]."
            ),
        },
        "policy_config": {
            "depth": int(parsed.depth),
            "temperature": float(parsed.temperature),
            "split_type": str(parsed.split_type),
            "leaf_type": str(parsed.leaf_type),
            "small_training_episodes": int(parsed.training_episodes_small),
            "large_training_episodes": int(parsed.training_episodes_large),
            "es_population": int(parsed.es_population),
            "sigma_init": float(parsed.sigma_init),
            "train_seed_batch": int(parsed.train_seed_batch),
        },
        "instance_results": {},
        "instance_table": [],
    }

    for reference_name in parsed.instance_names:
        result = _instance_result(parsed, reference_name)
        payload["instance_results"][reference_name] = result
        payload["instance_table"].append(result["summary"])
        aggregate_rows = payload["instance_table"]
        payload["aggregate"] = {
            "num_instances": len(aggregate_rows),
            "beats_best_repo_heuristic_count": int(
                sum(row["gap_vs_repo_best"] > 0.0 for row in aggregate_rows)
            ),
            "beats_best_published_heuristic_count": int(
                sum(row["gap_vs_published_best_heuristic"] > 0.0 for row in aggregate_rows)
            ),
            "beats_published_ppo_count": int(
                sum(row["gap_vs_published_ppo"] > 0.0 for row in aggregate_rows)
            ),
            "mean_gap_vs_repo_best": float(
                np.mean([row["gap_vs_repo_best"] for row in aggregate_rows])
            ),
            "mean_gap_vs_published_ppo": float(
                np.mean([row["gap_vs_published_ppo"] for row in aggregate_rows])
            ),
        }
        _write_progress(
            payload,
            output_json=parsed.output_json,
            output_markdown=parsed.output_markdown,
        )

    print(dumps_json(payload))
    print()
    print(_report_markdown(payload))


if __name__ == "__main__":
    main()
