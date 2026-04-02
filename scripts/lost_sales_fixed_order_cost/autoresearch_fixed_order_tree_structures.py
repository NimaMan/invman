import argparse
import json
import statistics
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policies.registry import apply_policy_name, make_soft_tree_policy_name
from invman.problems.lost_sales_fixed_order_cost.reference_instances import build_reference_args


BUDGETS = {
    "screening": {
        "training_episodes": 300,
        "es_population": 8,
        "horizon": 1500,
        "eval_horizon": 20000,
        "eval_seeds": 2,
    },
    "full": {
        "training_episodes": 2000,
        "es_population": 10,
        "horizon": 2000,
        "eval_horizon": 50000,
        "eval_seeds": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare candidate tree policy structures on the canonical fixed-order-cost benchmark."
    )
    parser.add_argument("--run_tag", default="fixed_cost_tree_search", help="Namespace used for outputs.")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--reference", default="lit_pois_mu5_l4_p4_k5", help="Named fixed-order-cost reference instance.")
    parser.add_argument("--tree_depths", nargs="+", type=int, default=[1, 2, 3], help="Tree depths to compare.")
    parser.add_argument(
        "--tree_split_types",
        nargs="+",
        choices=["oblique", "axis_aligned"],
        default=["oblique"],
        help="Tree split structures to compare.",
    )
    parser.add_argument(
        "--tree_leaf_types",
        nargs="+",
        choices=["constant", "linear"],
        default=["constant", "linear"],
        help="Tree leaf output types to compare.",
    )
    parser.add_argument(
        "--tree_temperatures",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5],
        help="Tree split temperatures to compare.",
    )
    parser.add_argument(
        "--sigma_inits",
        nargs="+",
        type=float,
        default=[2.0, 5.0],
        help="Initial CMA sigma values to compare.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit training seeds. When omitted, uses --seed as a single-element list.",
    )
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse existing per-run JSONs when present.")
    return parser.parse_args()


def _prepare_args(parsed, root, split_type, leaf_type, depth, temperature, sigma_init, seed):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(parsed.reference)
    args.problem = "lost_sales_fixed_order_cost"
    args.policy_name = make_soft_tree_policy_name(
        depth=depth,
        temperature=temperature,
        split_type=split_type,
        leaf_type=leaf_type,
    )
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.sigma_init = sigma_init
    args.seed = int(seed)
    args.mp_num_processors = parsed.mp_num_processors
    args.same_seed = parsed.same_seed
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = (
        f"{parsed.run_tag}_{parsed.budget}_{args.policy_name}_"
        f"sig{str(sigma_init).replace('.', 'p')}_seed{int(seed)}"
    )
    return args


def _result_path(args):
    return Path(args.results_dir) / f"{args.experiment_name}.json"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    result_path = _result_path(args)
    if reuse_existing and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8")), result_path
    return run_experiment(args)


def _summarize_result(payload, *, sigma_init, seed):
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    heuristic_cost = min(
        summary["mean_cost"]
        for summary in payload["evaluation"]["heuristics"].values()
        if isinstance(summary, dict) and "mean_cost" in summary
    )
    return {
        "experiment_name": payload["experiment_name"],
        "policy_architecture": payload["policy_architecture"],
        "tree_split_type": payload["tree_split_type"],
        "tree_leaf_type": payload["tree_leaf_type"],
        "tree_depth": payload["tree_depth"],
        "tree_temperature": payload["tree_temperature"],
        "sigma_init": float(sigma_init),
        "seed": int(seed),
        "learned_mean_cost": learned_cost,
        "best_heuristic_cost": heuristic_cost,
        "heuristic_gap": learned_cost - heuristic_cost,
        "results_file": payload.get("results_file"),
    }


def _aggregate_key(item):
    return (
        item["policy_architecture"],
        item["tree_split_type"],
        item["tree_leaf_type"],
        item["tree_depth"],
        item["tree_temperature"],
        item["sigma_init"],
    )


def _aggregate_results(results):
    grouped = {}
    for item in results:
        grouped.setdefault(_aggregate_key(item), []).append(item)

    aggregates = []
    for key, items in grouped.items():
        learned_costs = [item["learned_mean_cost"] for item in items]
        heuristic_gaps = [item["heuristic_gap"] for item in items]
        aggregates.append(
            {
                "policy_architecture": key[0],
                "tree_split_type": key[1],
                "tree_leaf_type": key[2],
                "tree_depth": key[3],
                "tree_temperature": key[4],
                "sigma_init": key[5],
                "num_seeds": len(items),
                "seed_list": [item["seed"] for item in items],
                "mean_learned_cost": float(statistics.fmean(learned_costs)),
                "median_learned_cost": float(statistics.median(learned_costs)),
                "best_learned_cost": float(min(learned_costs)),
                "worst_learned_cost": float(max(learned_costs)),
                "cost_range": float(max(learned_costs) - min(learned_costs)),
                "mean_heuristic_gap": float(statistics.fmean(heuristic_gaps)),
                "detailed_runs": [
                    {
                        "seed": item["seed"],
                        "learned_mean_cost": item["learned_mean_cost"],
                        "heuristic_gap": item["heuristic_gap"],
                        "results_file": item["results_file"],
                    }
                    for item in sorted(items, key=lambda run: run["seed"])
                ],
            }
        )

    aggregates.sort(key=lambda item: (item["median_learned_cost"], item["mean_learned_cost"], item["cost_range"]))
    return aggregates


def _render_markdown(summary):
    lines = [
        "# Fixed-Cost Tree Structure Autoresearch",
        "",
        f"Run tag: `{summary['run_tag']}`",
        f"Budget: `{summary['budget']}`",
        f"Reference: `{summary['reference']}`",
        f"Seeds: `{summary['seeds']}`",
        "",
        "## Aggregate Stability Summary",
        "",
        "| Architecture | Depth | Split | Leaf | Temp | Sigma | Seeds | Mean | Median | Best | Worst | Range | Mean gap |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["aggregates"]:
        lines.append(
            f"| `{item['policy_architecture']}` | `{item['tree_depth']}` | `{item['tree_split_type']}` | "
            f"`{item['tree_leaf_type']}` | `{item['tree_temperature']}` | `{item['sigma_init']}` | "
            f"`{item['num_seeds']}` | `{item['mean_learned_cost']:.5f}` | `{item['median_learned_cost']:.5f}` | "
            f"`{item['best_learned_cost']:.5f}` | `{item['worst_learned_cost']:.5f}` | "
            f"`{item['cost_range']:.5f}` | `{item['mean_heuristic_gap']:.5f}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(parsed.seed)] if parsed.seeds is None else [int(seed) for seed in parsed.seeds]

    results = []
    for split_type in parsed.tree_split_types:
        for leaf_type in parsed.tree_leaf_types:
            for depth in parsed.tree_depths:
                for temperature in parsed.tree_temperatures:
                    for sigma_init in parsed.sigma_inits:
                        for seed in seeds:
                            args = _prepare_args(parsed, root, split_type, leaf_type, depth, temperature, sigma_init, seed)
                            payload, results_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
                            payload["results_file"] = str(results_path)
                            results.append(_summarize_result(payload, sigma_init=sigma_init, seed=seed))

    results.sort(key=lambda item: (item["learned_mean_cost"], item["heuristic_gap"], item["seed"]))
    aggregates = _aggregate_results(results)
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "seeds": seeds,
        "tree_depths": parsed.tree_depths,
        "tree_split_types": parsed.tree_split_types,
        "tree_leaf_types": parsed.tree_leaf_types,
        "tree_temperatures": parsed.tree_temperatures,
        "sigma_inits": parsed.sigma_inits,
        "detailed_results": results,
        "aggregates": aggregates,
        "best_result": results[0] if results else None,
        "best_aggregate": aggregates[0] if aggregates else None,
    }

    summary_path = root / f"fixed_order_tree_search_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = root / f"fixed_order_tree_search_{parsed.budget}.md"
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
