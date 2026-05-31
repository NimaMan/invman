"""
Sweep candidate soft-tree policy variants across dual-sourcing reference instances.

OBJECTIVE
    Evaluate a roster of named soft-tree policy variants on one or more
    Gijsbrechts-2022 Figure-9 dual-sourcing instances, ranking each by its
    optimality gap vs the strongest heuristic (capped_dual_index). Dual sourcing
    is soft_tree-ONLY after the Rust migration, so the roster is soft-tree
    structures over the dual-index / capped-dual-index / base-surge control bases
    (the deleted dense linear/nn variants are dropped).

ALGORITHM
    For each (reference x policy):
      1. Build args from the Rust reference instance; train via CMA-ES.
      2. Evaluate learned mean cost over eval_seeds at eval_horizon.
      3. Compute the best heuristic cost from the Rust search bindings (once per
         reference) and the learned-policy gap.
    Write a JSON summary sorted by (reference, gap).
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


# Soft-tree-only roster. Covers the promoted axis-constant small-cap structure,
# oblique linear/constant leaves, and depth 2/3 over the capped-dual-index and
# base-surge control bases.
DEFAULT_POLICIES = [
    "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
    "soft_tree_capped_dual_index_delta_smallcap_targets",
    make_soft_tree_policy_name(depth=2, temperature=0.25, split_type="oblique", leaf_type="linear", action_adapter="capped_dual_index_targets"),
    make_soft_tree_policy_name(depth=3, temperature=0.25, split_type="oblique", leaf_type="linear", action_adapter="capped_dual_index_targets"),
    make_soft_tree_policy_name(depth=2, temperature=0.25, split_type="oblique", leaf_type="constant", action_adapter="capped_dual_index_targets"),
    make_soft_tree_policy_name(depth=2, temperature=0.25, split_type="oblique", leaf_type="linear", action_adapter="base_surge_targets"),
    make_soft_tree_policy_name(depth=3, temperature=0.25, split_type="oblique", leaf_type="linear", action_adapter="base_surge_targets"),
    make_soft_tree_policy_name(depth=2, temperature=0.25, split_type="oblique", leaf_type="constant", action_adapter="base_surge_targets"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep candidate dual-sourcing soft-tree policy variants on named reference instances.")
    parser.add_argument(
        "--references",
        nargs="+",
        default=["dual_l3_ce105", "dual_l3_ce110"],
        help="Reference instances to evaluate.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Soft-tree policy names to train and compare.",
    )
    parser.add_argument("--run_tag", default="dual_l3_policy_variants")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--training_episodes", type=int, default=300)
    parser.add_argument("--es_population", type=int, default=8)
    parser.add_argument("--es_population_sampling", default="fixed")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--eval_horizon", type=int, default=5000)
    parser.add_argument("--eval_seeds", type=int, default=2)
    parser.add_argument("--sigma_init", type=float, default=3.0)
    parser.add_argument("--rollout_backend", default="rust")
    return parser.parse_args()


def configure_args(parsed, reference_name: str, policy_name: str, root: Path):
    args = lib.build_reference_args(reference_name)
    args.problem = "dual_sourcing"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = parsed.training_episodes
    args.es_population = parsed.es_population
    args.es_population_sampling = parsed.es_population_sampling
    args.horizon = parsed.horizon
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = parsed.sigma_init
    args.policy_name = policy_name
    apply_policy_name(args)
    args.rollout_backend = parsed.rollout_backend
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{reference_name}_{args.policy_name}"
    return args


def summarize_payload(payload, results_path: Path, reference_name, best_heuristic_name, best_heuristic_cost):
    learned_cost = lib.learned_cost_of(payload)
    gap_pct = None if best_heuristic_cost is None else 100.0 * (learned_cost / best_heuristic_cost - 1.0)
    return {
        "reference": reference_name,
        "experiment_name": payload["experiment_name"],
        "policy_name": payload["policy_name"],
        "policy_architecture": payload["policy_architecture"],
        "action_adapter": payload["action_adapter"],
        "learned_mean_cost": learned_cost,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "gap_pct_vs_best_heuristic": gap_pct,
        "results_file": str(results_path),
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    summary = []
    for reference_name in parsed.references:
        probe_args = lib.build_reference_args(reference_name)
        heuristics = lib.evaluate_default_heuristics(probe_args)
        best_heuristic_name, best_heuristic_cost = lib.best_heuristic(heuristics)
        for policy_name in parsed.policies:
            args = configure_args(parsed, reference_name, policy_name, root)
            payload, results_path = run_experiment(args)
            row = summarize_payload(payload, results_path, reference_name, best_heuristic_name, best_heuristic_cost)
            summary.append(row)
            gap_str = "n/a" if row["gap_pct_vs_best_heuristic"] is None else f"{row['gap_pct_vs_best_heuristic']:.4f}%"
            print(
                f"{reference_name} {policy_name} "
                f"learned={row['learned_mean_cost']:.6f} "
                f"best_heur={best_heuristic_name}={best_heuristic_cost:.6f} "
                f"gap={gap_str}"
            )

    summary.sort(key=lambda row: (row["reference"], row["gap_pct_vs_best_heuristic"] if row["gap_pct_vs_best_heuristic"] is not None else float("inf")))
    output_path = root / "screening_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
