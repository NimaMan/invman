"""
Single-policy autoresearch runner for the dual-sourcing benchmark.

OBJECTIVE
    Train one soft-tree CMA-ES policy on a named Gijsbrechts-2022 Figure-9
    dual-sourcing instance and log its cost + optimality gap vs the strongest
    heuristic (capped_dual_index). Dual sourcing routes entirely through the Rust
    extension after the Python-cleanup migration, and its policy backbone is
    soft_tree ONLY, so the search surface is the soft-tree structure (depth,
    temperature, oblique/axis-aligned split, constant/linear leaf) over a control
    adapter.

ALGORITHM
    1. Build args from the Rust reference instance (dual_sourcing_benchmark_lib).
    2. Compose the soft-tree policy name from CLI structure flags; apply CMA-ES
       budget (screening or full).
    3. run_experiment trains via invman_rust.dual_sourcing_soft_tree_rollout and
       evaluates mean cost over eval_seeds at eval_horizon.
    4. Heuristic baselines come from the Rust search bindings (NOT the experiment
       payload, which carries an empty heuristics block for dual sourcing).
    5. Append a TSV row: learned cost, best heuristic, gap, gap%.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dual_sourcing_benchmark_lib as lib

from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch-style loop for the dual-sourcing benchmark.")
    parser.add_argument("--run_tag", default="dual_sourcing_autoresearch")
    parser.add_argument("--budget", choices=sorted(lib.COMMON_BUDGET), default="screening")
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


def main():
    parsed = parse_args()
    args = lib.build_reference_args(parsed.reference)
    budget = lib.get_budget_config(parsed.budget)
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
    lib._apply_budget(args, budget)
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
    args.mp_num_processors = parsed.mp_num_processors
    args.experiment_name = f"{parsed.run_tag}_{parsed.budget}_{args.policy_name}"

    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    results_tsv = root / "results.tsv"
    root.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["commit", "experiment_name", "reference", "budget", "policy_architecture", "mean_cost", "best_heuristic", "best_heuristic_name", "heuristic_gap", "heuristic_gap_pct", "description"])

    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    payload, results_path = run_experiment(args)
    learned_cost = lib.learned_cost_of(payload)

    # Heuristic baselines come from the Rust search bindings: the dual-sourcing
    # experiment payload carries an empty heuristics block.
    heuristics = lib.evaluate_default_heuristics(args)
    best_heuristic_name, best_heuristic_cost = lib.best_heuristic(heuristics)
    if best_heuristic_cost is None:
        gap = gap_pct = None
    else:
        gap = learned_cost - best_heuristic_cost
        gap_pct = 100.0 * (learned_cost / best_heuristic_cost - 1.0)

    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            _git_short_commit(PACKAGE_ROOT),
            args.experiment_name,
            parsed.reference,
            parsed.budget,
            payload["policy_architecture"],
            f"{learned_cost:.6f}",
            "" if best_heuristic_cost is None else f"{best_heuristic_cost:.6f}",
            best_heuristic_name or "",
            "" if gap is None else f"{gap:.6f}",
            "" if gap_pct is None else f"{gap_pct:.4f}",
            parsed.description,
        ])
    print(json.dumps({
        "results_json": str(results_path),
        "reference": parsed.reference,
        "learned_mean_cost": learned_cost,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "heuristic_gap_pct": gap_pct,
        "heuristics": heuristics,
    }, indent=2))


if __name__ == "__main__":
    main()
