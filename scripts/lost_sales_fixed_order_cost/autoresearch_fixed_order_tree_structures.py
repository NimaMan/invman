import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
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
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    return parser.parse_args()


def _prepare_args(parsed, root, split_type, leaf_type, depth, temperature, sigma_init):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(parsed.reference)
    args.problem = "lost_sales_fixed_order_cost"
    args.policy_type = "soft_tree"
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.tree_depth = depth
    args.tree_temperature = temperature
    args.tree_split_type = split_type
    args.tree_leaf_type = leaf_type
    args.sigma_init = sigma_init
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
    args.experiment_name = (
        f"{parsed.run_tag}_{parsed.budget}_{split_type}_{leaf_type}_"
        f"d{depth}_t{str(temperature).replace('.', 'p')}_s{str(sigma_init).replace('.', 'p')}"
    )
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
        "tree_split_type": payload["tree_split_type"],
        "tree_leaf_type": payload["tree_leaf_type"],
        "tree_depth": payload["tree_depth"],
        "tree_temperature": payload["tree_temperature"],
        "sigma_init": payload["evaluation"].get("sigma_init", None),
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
    for split_type in parsed.tree_split_types:
        for leaf_type in parsed.tree_leaf_types:
            for depth in parsed.tree_depths:
                for temperature in parsed.tree_temperatures:
                    for sigma_init in parsed.sigma_inits:
                        args = _prepare_args(parsed, root, split_type, leaf_type, depth, temperature, sigma_init)
                        payload, results_path = run_experiment(args)
                        payload["results_file"] = str(results_path)
                        result = _summarize_result(payload)
                        result["sigma_init"] = sigma_init
                        results.append(result)

    results.sort(key=lambda item: item["learned_mean_cost"])
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "tree_depths": parsed.tree_depths,
        "tree_split_types": parsed.tree_split_types,
        "tree_leaf_types": parsed.tree_leaf_types,
        "tree_temperatures": parsed.tree_temperatures,
        "sigma_inits": parsed.sigma_inits,
        "results": results,
        "best_result": results[0] if results else None,
    }

    summary_path = root / f"fixed_order_tree_search_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
