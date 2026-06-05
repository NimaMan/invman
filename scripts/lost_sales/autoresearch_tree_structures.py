import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name
from scripts.lost_sales.benchmark_canonical_suite import build_reference_args

DEFAULT_REFERENCE = "vanilla_l4_p4_poisson5"


BUDGETS = {
    "screening": {
        "training_episodes": 100,
        "es_population": 8,
        "horizon": 1000,
        "eval_horizon": 10000,
        "eval_seeds": 2,
    },
    "full": {
        "training_episodes": 2000,
        "es_population": 10,
        "horizon": 2000,
        "eval_horizon": int(1e5),
        "eval_seeds": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare candidate tree policy structures on the trusted lost-sales benchmark.")
    parser.add_argument("--run_tag", default="tree_structure_search", help="Namespace used for outputs.")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Named reference instance.")
    parser.add_argument("--tree_depths", nargs="+", type=int, default=[2, 3], help="Tree depths to compare.")
    parser.add_argument(
        "--tree_split_types",
        nargs="+",
        choices=["oblique", "axis_aligned"],
        default=["oblique", "axis_aligned"],
        help="Tree split structures to compare.",
    )
    parser.add_argument(
        "--tree_leaf_types",
        nargs="+",
        choices=["constant", "linear"],
        default=["constant"],
        help="Tree leaf output types to compare.",
    )
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--sigma_init", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    return parser.parse_args()


def _prepare_args(parsed, root, split_type, leaf_type, depth):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(parsed.reference)
    args.problem = "lost_sales"
    args.policy_name = make_soft_tree_policy_name(
        depth=depth,
        temperature=parsed.tree_temperature,
        split_type=split_type,
        leaf_type=leaf_type,
    )
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
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
    args.experiment_name = f"{parsed.run_tag}_{parsed.budget}_{args.policy_name}"
    return args


def _summarize_result(payload):
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    heuristic_costs = [
        summary["mean_cost"]
        for summary in payload["evaluation"]["heuristics"].values()
        if isinstance(summary, dict) and summary.get("mean_cost") is not None
    ]
    heuristic_cost = min(heuristic_costs) if heuristic_costs else None
    return {
        "experiment_name": payload["experiment_name"],
        "policy_architecture": payload["policy_architecture"],
        "tree_split_type": payload["tree_split_type"],
        "tree_leaf_type": payload["tree_leaf_type"],
        "tree_depth": payload["tree_depth"],
        "learned_mean_cost": learned_cost,
        "best_heuristic_cost": heuristic_cost,
        "heuristic_gap": None if heuristic_cost is None else learned_cost - heuristic_cost,
        "results_file": payload.get("results_file"),
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for split_type in parsed.tree_split_types:
        for leaf_type in parsed.tree_leaf_types:
            for depth in parsed.tree_depths:
                args = _prepare_args(parsed, root, split_type, leaf_type, depth)
                payload, results_path = run_experiment(args)
                payload["results_file"] = str(results_path)
                results.append(_summarize_result(payload))

    results.sort(key=lambda item: item["learned_mean_cost"])
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "tree_depths": parsed.tree_depths,
        "tree_split_types": parsed.tree_split_types,
        "tree_leaf_types": parsed.tree_leaf_types,
        "results": results,
        "best_result": results[0] if results else None,
    }

    summary_path = root / f"tree_structure_search_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
