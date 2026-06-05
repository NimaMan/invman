import argparse
import json
import sys
from copy import copy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name
from scripts.lost_sales.benchmark_canonical_suite import build_reference_args


POLICIES = [
    "linear_categorical_quantity_q20",
    "linear_sigmoid_direct_quantity",
    "linear_soft_gated_direct_quantity",
    "linear_hard_gated_direct_quantity",
    "linear_soft_gated_ordinal_quantity",
    "nn_categorical_quantity_q20",
    "nn_soft_gated_ordinal_quantity",
    "soft_tree_depth1_linear_leaf",
    "soft_tree_depth2_linear_leaf",
    "soft_tree_depth1_sigmoid_linear_leaf_q20",
    "soft_tree_depth2_sigmoid_linear_leaf_q20",
]

DEMAND_CASES = [
    ("poisson", {"demand_dist_name": "Poisson", "demand_rate": 5.0}),
    ("geometric", {"demand_dist_name": "Geometric", "demand_rate": 5.0}),
    (
        "mmpp2_positive",
        {
            "demand_dist_name": "MarkovModulatedPoisson2",
            "demand_rate": 5.0,
            "demand_lambda_low": 3.0,
            "demand_lambda_high": 7.0,
            "demand_p00": 0.9,
            "demand_p11": 0.9,
        },
    ),
    (
        "mmpp2_negative",
        {
            "demand_dist_name": "MarkovModulatedPoisson2",
            "demand_rate": 5.0,
            "demand_lambda_low": 3.0,
            "demand_lambda_high": 7.0,
            "demand_p00": 0.1,
            "demand_p11": 0.1,
        },
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the clean L=4 vanilla lost-sales demand policy suite.")
    parser.add_argument("--run_tag", default="lost_sales_l4_p4_demand_policy_suite_clean_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    return parser.parse_args()


def write_summary(summary, summary_json: Path, summary_md: Path):
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Lost-Sales L4/P4 Demand Policy Suite",
        "",
        f"- run_tag: `{summary['run_tag']}`",
        f"- training_episodes: `{summary['training_episodes']}`",
        f"- es_population: `{summary['es_population']}`",
        f"- training_horizon: `{summary['training_horizon']}`",
        f"- evaluation_horizon: `{summary['evaluation_horizon']}`",
        f"- evaluation_seeds: `{summary['evaluation_seeds']}`",
        "",
    ]
    for demand_case, payload in summary["demand_cases"].items():
        lines.append(f"## {demand_case}")
        heuristics = payload.get("heuristics", {})
        if heuristics:
            lines.extend(["", "| heuristic | mean_cost |", "| --- | ---: |"])
            for name, result in heuristics.items():
                lines.append(f"| {name} | {result['mean_cost']:.6f} |")
        results = payload.get("results", {})
        if results:
            lines.extend(["", "| policy | mean_cost | std_cost |", "| --- | ---: | ---: |"])
            for policy_name, result in results.items():
                lines.append(f"| {policy_name} | {result['mean_cost']:.6f} | {result['std_cost']:.6f} |")
        lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def build_args(parsed, results_dir: Path, logs_dir: Path, models_dir: Path, demand_case: str, demand_kwargs: dict, policy_name: str):
    args = build_reference_args("vanilla_l4_p4_poisson5")
    args.problem = "lost_sales"
    args.reference_instance = "vanilla_l4_p4_poisson5"
    args.seed = parsed.seed
    args.same_seed = False
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = 2000
    args.es_population = 64
    args.horizon = 2000
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = 5.0
    args.save_every = 1000
    args.max_order_size = 20
    for key, value in demand_kwargs.items():
        setattr(args, key, value)
    args.policy_name = policy_name
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.results_dir = str(results_dir)
    args.log_dir = str(logs_dir)
    args.trained_models_dir = str(models_dir)
    args.experiment_name = f"{parsed.run_tag}_{demand_case}_{policy_name}"
    return args


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "benchmarks" / parsed.run_tag
    results_dir = root / "results"
    logs_dir = root / "logs"
    models_dir = root / "models"
    summary_json = root / "lost_sales_l4_p4_demand_policy_suite.json"
    summary_md = root / "lost_sales_l4_p4_demand_policy_suite.md"

    for directory in (results_dir, logs_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_tag": parsed.run_tag,
        "problem": "lost_sales",
        "reference_base": "vanilla_l4_p4_poisson5",
        "lead_time": 4,
        "shortage_cost": 4.0,
        "training_episodes": 2000,
        "es_population": 64,
        "training_horizon": 2000,
        "evaluation_horizon": parsed.eval_horizon,
        "evaluation_seeds": parsed.eval_seeds,
        "policies": list(POLICIES),
        "demand_cases": {},
    }
    write_summary(summary, summary_json, summary_md)

    for demand_case, demand_kwargs in DEMAND_CASES:
        demand_summary = {"params": dict(demand_kwargs), "heuristics": {}, "results": {}}
        summary["demand_cases"][demand_case] = demand_summary
        write_summary(summary, summary_json, summary_md)

        for idx, policy_name in enumerate(POLICIES):
            args = build_args(parsed, results_dir, logs_dir, models_dir, demand_case, demand_kwargs, policy_name)
            payload, result_path = run_experiment(args)
            learned = copy(payload["evaluation"]["learned_policy"])
            learned["result_path"] = str(result_path)
            demand_summary["results"][policy_name] = learned
            if idx == 0:
                demand_summary["heuristics"] = payload["evaluation"]["heuristics"]
            write_summary(summary, summary_json, summary_md)


if __name__ == "__main__":
    main()
