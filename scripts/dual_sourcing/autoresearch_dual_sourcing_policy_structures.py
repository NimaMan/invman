import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policies.registry import apply_policy_name, make_soft_tree_policy_name
from invman.problems.dual_sourcing.reference_instances import build_reference_args


BUDGETS = {
    "screening": {
        "training_episodes": 300,
        "es_population": 8,
        "horizon": 1000,
        "eval_horizon": 5000,
        "eval_seeds": 2,
    },
    "full": {
        "training_episodes": 1500,
        "es_population": 128,
        "es_population_sampling": "categorical",
        "es_population_candidates": [32, 64, 96, 128],
        "es_population_probabilities": [0.05, 0.15, 0.25, 0.55],
        "horizon": 2000,
        "eval_horizon": 10000,
        "eval_seeds": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare candidate learned policy structures on the primary dual-sourcing benchmark.")
    parser.add_argument("--run_tag", default="dual_sourcing_policy_search", help="Namespace used for outputs.")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--reference", default="dual_l4_ce110", help="Named dual-sourcing reference instance.")
    parser.add_argument(
        "--action_adapters",
        nargs="+",
        default=[
            "identity",
            "single_index_targets",
            "dual_index_targets",
            "capped_dual_index_targets",
            "base_surge_targets",
        ],
        help="Structured action adapters to compare.",
    )
    parser.add_argument("--tree_depths", nargs="+", type=int, default=[2], help="Tree depths to compare.")
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--tree_leaf_type", choices=["constant", "linear"], default="linear")
    parser.add_argument("--sigma_init", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    return parser.parse_args()


def _prepare_args(parsed, root, action_adapter, depth):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(parsed.reference)
    args.problem = "dual_sourcing"
    args.policy_name = make_soft_tree_policy_name(
        depth=depth,
        temperature=parsed.tree_temperature,
        split_type=parsed.tree_split_type,
        leaf_type=parsed.tree_leaf_type,
        action_adapter=action_adapter,
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
    args.es_population_sampling = budget.get("es_population_sampling", "fixed")
    args.es_population_candidates = budget.get("es_population_candidates")
    args.es_population_probabilities = budget.get("es_population_probabilities")
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
    heuristic_cost = min(
        summary["mean_cost"]
        for summary in payload["evaluation"]["heuristics"].values()
        if isinstance(summary, dict) and "mean_cost" in summary
    )
    return {
        "experiment_name": payload["experiment_name"],
        "policy_architecture": payload["policy_architecture"],
        "action_adapter": payload.get("action_adapter", "identity"),
        "tree_depth": payload["tree_depth"],
        "tree_split_type": payload["tree_split_type"],
        "tree_leaf_type": payload["tree_leaf_type"],
        "learned_mean_cost": learned_cost,
        "best_heuristic_cost": heuristic_cost,
        "heuristic_gap": learned_cost - heuristic_cost,
        "results_file": payload.get("results_file"),
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for action_adapter in parsed.action_adapters:
        for depth in parsed.tree_depths:
            args = _prepare_args(parsed, root, action_adapter, depth)
            payload, results_path = run_experiment(args)
            payload["results_file"] = str(results_path)
            results.append(_summarize_result(payload))

    results.sort(key=lambda item: item["learned_mean_cost"])
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "action_adapters": parsed.action_adapters,
        "tree_depths": parsed.tree_depths,
        "tree_split_type": parsed.tree_split_type,
        "tree_leaf_type": parsed.tree_leaf_type,
        "results": results,
        "best_result": results[0] if results else None,
    }

    summary_path = root / f"dual_sourcing_policy_search_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
