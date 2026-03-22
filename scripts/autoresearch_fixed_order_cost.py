import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    build_reference_args,
    get_reference_instance,
)


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
        description="Autoresearch-style benchmark harness for the canonical fixed-order-cost instance."
    )
    parser.add_argument("--run_tag", default="fixed_order_cost_autoresearch", help="Autoresearch run namespace.")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--description", required=True, help="Short description of the policy change being tested.")
    parser.add_argument("--reference", default="lit_pois_mu5_l4_p4_k5", help="Named fixed-order-cost reference instance.")
    parser.add_argument("--policy_type", choices=["linear", "nn", "soft_tree"], default="soft_tree")
    parser.add_argument("--policy_head", default="categorical_quantity", help="Policy head for nn/linear backbones.")
    parser.add_argument("--rollout_backend", choices=["python", "rust"], default="python")
    parser.add_argument("--training_method", choices=["cma", "es", "ga", "xnes"], default="cma")
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--tree_leaf_type", choices=["constant", "linear"], default="linear")
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--activation", choices=["selu", "gelu", "relu"], default="selu")
    parser.add_argument("--sigma_init", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    parser.add_argument("--max_order_size", type=int, default=None, help="Override the reference max order size.")
    return parser.parse_args()


def _git_short_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root.parent), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _results_paths(project_root: Path, run_tag: str):
    root = project_root / "outputs" / "autoresearch" / run_tag
    return {
        "root": root,
        "results": root / "results",
        "logs": root / "logs",
        "models": root / "models",
        "tsv": root / "results.tsv",
    }


def _ensure_results_tsv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "commit",
                "experiment_name",
                "reference",
                "budget",
                "policy_architecture",
                "mean_cost",
                "best_heuristic",
                "heuristic_gap",
                "status",
                "description",
            ]
        )


def _best_prior_cost(tsv_path: Path):
    if not tsv_path.exists():
        return None
    best_cost = None
    with tsv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["status"] != "keep":
                continue
            cost = float(row["mean_cost"])
            if best_cost is None or cost < best_cost:
                best_cost = cost
    return best_cost


def _append_results_row(tsv_path: Path, row):
    with tsv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def _configure_args(parsed):
    args = build_reference_args(parsed.reference)
    budget = BUDGETS[parsed.budget]
    reference = get_reference_instance(parsed.reference)
    args.problem = "lost_sales_fixed_order_cost"
    args.policy_type = parsed.policy_type
    args.policy_head = parsed.policy_head
    args.rollout_backend = parsed.rollout_backend
    args.training_method = parsed.training_method
    args.tree_depth = parsed.tree_depth
    args.tree_temperature = parsed.tree_temperature
    args.tree_split_type = parsed.tree_split_type
    args.tree_leaf_type = parsed.tree_leaf_type
    args.hidden_dim = parsed.hidden_dim
    args.activation = parsed.activation
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
    args.mp_num_processors = parsed.mp_num_processors
    args.same_seed = parsed.same_seed
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]
    if parsed.max_order_size is not None:
        args.max_order_size = parsed.max_order_size
    if args.policy_type == "soft_tree":
        args.rollout_backend = "rust"
    args.reference_instance_name = reference["name"]
    return args


def main():
    parsed = parse_args()
    project_root = PACKAGE_ROOT
    paths = _results_paths(project_root, parsed.run_tag)
    _ensure_results_tsv(paths["tsv"])

    args = _configure_args(parsed)
    args.results_dir = str(paths["results"])
    args.log_dir = str(paths["logs"])
    args.trained_models_dir = str(paths["models"])
    args.experiment_name = (
        f"{parsed.run_tag}_{parsed.budget}_{args.policy_type}_{args.policy_head}"
        if args.policy_type != "soft_tree"
        else (
            f"{parsed.run_tag}_{parsed.budget}_soft_tree_"
            f"{args.tree_split_type}_{args.tree_leaf_type}_d{args.tree_depth}_"
            f"t{str(args.tree_temperature).replace('.', 'p')}_"
            f"s{str(args.sigma_init).replace('.', 'p')}"
        )
    )

    payload, results_path = run_experiment(args)
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    heuristic_results = payload["evaluation"]["heuristics"]
    best_heuristic_cost = min(
        heuristic_summary["mean_cost"]
        for heuristic_summary in heuristic_results.values()
        if isinstance(heuristic_summary, dict) and "mean_cost" in heuristic_summary
    )
    heuristic_gap = learned_cost - best_heuristic_cost
    prior_best = _best_prior_cost(paths["tsv"])
    status = "keep" if prior_best is None or learned_cost < prior_best else "discard"

    row = [
        _git_short_commit(project_root),
        args.experiment_name,
        parsed.reference,
        parsed.budget,
        payload["policy_architecture"],
        f"{learned_cost:.6f}",
        f"{best_heuristic_cost:.6f}",
        f"{heuristic_gap:.6f}",
        status,
        parsed.description,
    ]
    _append_results_row(paths["tsv"], row)

    summary = {
        "results_tsv": str(paths["tsv"]),
        "results_json": str(results_path),
        "status": status,
        "payload": payload,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
