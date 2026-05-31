"""
Soft-tree structure screening for the dual-sourcing benchmark.

OBJECTIVE
    Find which soft-tree policy STRUCTURE performs best on a dual-sourcing
    Figure-9 instance, given that dual sourcing is soft_tree-ONLY after the
    Rust migration. We sweep the structure axes the CMA-ES search controls:
    control adapter (identity / single- / dual- / capped-dual-index / base-surge
    target bases), tree depth, oblique vs axis-aligned splits, and constant vs
    linear leaves. The goal is a learned soft-tree policy whose optimality gap is
    competitive with the strongest heuristic capped_dual_index (~0-0.11%) and
    beats the paper's A3C baseline (~0.5-1.85%).

ALGORITHM
    For each (action_adapter x depth) combination:
      1. Build args from the Rust reference instance; compose the soft-tree name.
      2. Train with CMA-ES (run_experiment -> dual_sourcing_soft_tree_rollout).
      3. Evaluate learned mean cost over eval_seeds at eval_horizon.
      4. Compute the best heuristic cost from the Rust search bindings and the gap.
    Rank candidates by learned cost (equivalently gap vs best heuristic) and write
    a ranked JSON summary. The best 1-3 structures are promoted to the benchmark
    suite. Heuristics are computed once per instance (shared across candidates).
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Compare candidate soft-tree structures on a dual-sourcing benchmark instance.")
    parser.add_argument("--run_tag", default="dual_sourcing_policy_search", help="Namespace used for outputs.")
    parser.add_argument("--budget", choices=sorted(lib.COMMON_BUDGET), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--reference", default="dual_l4_ce110", help="Named dual-sourcing reference instance.")
    parser.add_argument(
        "--action_adapters",
        nargs="+",
        default=[
            "identity",
            "single_index_targets",
            "dual_index_targets",
            "capped_dual_index_targets",
            "capped_dual_index_delta_smallcap_targets",
            "base_surge_targets",
        ],
        help="Structured action adapters to compare.",
    )
    parser.add_argument("--tree_depths", nargs="+", type=int, default=[1, 2], help="Tree depths to compare.")
    parser.add_argument("--tree_temperature", type=float, default=0.25)
    parser.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--tree_leaf_type", choices=["constant", "linear"], default="linear")
    parser.add_argument("--sigma_init", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within an ES batch.")
    return parser.parse_args()


def _prepare_args(parsed, root, action_adapter, depth):
    budget = lib.get_budget_config(parsed.budget)
    args = lib.build_reference_args(parsed.reference)
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
    lib._apply_budget(args, budget)
    args.sigma_init = parsed.sigma_init
    args.seed = parsed.seed
    args.mp_num_processors = parsed.mp_num_processors
    args.same_seed = parsed.same_seed
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{parsed.run_tag}_{parsed.budget}_{parsed.reference}_{args.policy_name}"
    return args


def _summarize_result(payload, best_heuristic_name, best_heuristic_cost):
    learned_cost = lib.learned_cost_of(payload)
    gap = None if best_heuristic_cost is None else learned_cost - best_heuristic_cost
    gap_pct = None if best_heuristic_cost is None else 100.0 * (learned_cost / best_heuristic_cost - 1.0)
    return {
        "experiment_name": payload["experiment_name"],
        "policy_name": payload["policy_name"],
        "policy_architecture": payload["policy_architecture"],
        "action_adapter": payload.get("action_adapter", "identity"),
        "tree_depth": payload["tree_depth"],
        "tree_split_type": payload["tree_split_type"],
        "tree_leaf_type": payload["tree_leaf_type"],
        "learned_mean_cost": learned_cost,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "heuristic_gap": gap,
        "gap_pct_vs_best_heuristic": gap_pct,
        "results_file": payload.get("results_file"),
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    # Heuristic baselines for this instance are identical across structures, so
    # compute them once via the Rust search bindings.
    probe_args = lib.build_reference_args(parsed.reference)
    heuristics = lib.evaluate_default_heuristics(probe_args)
    best_heuristic_name, best_heuristic_cost = lib.best_heuristic(heuristics)

    results = []
    for action_adapter in parsed.action_adapters:
        for depth in parsed.tree_depths:
            args = _prepare_args(parsed, root, action_adapter, depth)
            payload, results_path = run_experiment(args)
            payload["results_file"] = str(results_path)
            results.append(_summarize_result(payload, best_heuristic_name, best_heuristic_cost))
            row = results[-1]
            print(
                f"{parsed.reference} adapter={action_adapter} depth={depth} "
                f"learned={row['learned_mean_cost']:.4f} "
                f"best_heur={best_heuristic_name}={best_heuristic_cost:.4f} "
                f"gap={row['gap_pct_vs_best_heuristic']:.4f}%"
            )

    results.sort(key=lambda item: item["learned_mean_cost"])
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "action_adapters": parsed.action_adapters,
        "tree_depths": parsed.tree_depths,
        "tree_split_type": parsed.tree_split_type,
        "tree_leaf_type": parsed.tree_leaf_type,
        "heuristics": heuristics,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "published_optimality_gap_pct": lib._reference_instance(parsed.reference).get("published_optimality_gap_pct", {}),
        "results": results,
        "best_result": results[0] if results else None,
    }

    summary_path = root / f"dual_sourcing_policy_search_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
