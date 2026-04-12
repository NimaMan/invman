import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policies.registry import apply_policy_name, make_soft_tree_policy_name
from invman.problems.dual_sourcing.reference_instances import build_reference_args


BUDGETS = {
    "screening": {"training_episodes": 300, "es_population": 8, "horizon": 1000, "eval_horizon": 5000, "eval_seeds": 2},
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
    parser = argparse.ArgumentParser(description="Autoresearch-style loop for the dual-sourcing benchmark.")
    parser.add_argument("--run_tag", default="dual_sourcing_autoresearch")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    parser.add_argument("--description", required=True)
    parser.add_argument("--reference", default="dual_l4_ce110")
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--tree_leaf_type", choices=["constant", "linear"], default="linear")
    parser.add_argument(
        "--action_adapter",
        default="identity",
        help="Structured soft-tree action adapter to use for dual sourcing.",
    )
    parser.add_argument("--sigma_init", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    return parser.parse_args()


def _git_short_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip()


def _apply_budget(args, budget):
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.es_population_sampling = budget.get("es_population_sampling", "fixed")
    args.es_population_candidates = budget.get("es_population_candidates")
    args.es_population_probabilities = budget.get("es_population_probabilities")
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]


def main():
    parsed = parse_args()
    args = build_reference_args(parsed.reference)
    budget = BUDGETS[parsed.budget]
    args.problem = "dual_sourcing"
    args.policy_name = make_soft_tree_policy_name(
        depth=parsed.tree_depth,
        temperature=parsed.tree_temperature,
        split_type=parsed.tree_split_type,
        leaf_type=parsed.tree_leaf_type,
        action_adapter=parsed.action_adapter,
    )
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
    args.mp_num_processors = parsed.mp_num_processors
    _apply_budget(args, budget)
    args.experiment_name = f"{parsed.run_tag}_{parsed.budget}_{args.policy_name}"

    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    results_tsv = root / "results.tsv"
    root.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["commit", "experiment_name", "reference", "budget", "policy_architecture", "mean_cost", "best_heuristic", "heuristic_gap", "description"])

    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    payload, results_path = run_experiment(args)
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    best_heuristic_cost = min(value["mean_cost"] for value in payload["evaluation"]["heuristics"].values())
    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            _git_short_commit(PACKAGE_ROOT),
            args.experiment_name,
            parsed.reference,
            parsed.budget,
            payload["policy_architecture"],
            f"{learned_cost:.6f}",
            f"{best_heuristic_cost:.6f}",
            f"{learned_cost - best_heuristic_cost:.6f}",
            parsed.description,
        ])
    print(json.dumps({"results_json": str(results_path), "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
