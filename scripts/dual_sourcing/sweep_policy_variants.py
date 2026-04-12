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


DEFAULT_POLICIES = [
    "linear_base_surge_targets",
    "nn_base_surge_targets",
    "linear_capped_dual_index_delta_targets",
    "nn_capped_dual_index_delta_targets",
    make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
        action_adapter="base_surge_targets",
    ),
    make_soft_tree_policy_name(
        depth=3,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
        action_adapter="base_surge_targets",
    ),
    make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
        action_adapter="capped_dual_index_targets",
    ),
    make_soft_tree_policy_name(
        depth=3,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
        action_adapter="capped_dual_index_targets",
    ),
    make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
        action_adapter="base_surge_targets",
    ),
    make_soft_tree_policy_name(
        depth=3,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
        action_adapter="base_surge_targets",
    ),
    make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
        action_adapter="capped_dual_index_targets",
    ),
    make_soft_tree_policy_name(
        depth=3,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
        action_adapter="capped_dual_index_targets",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep candidate dual-sourcing policy variants on named reference instances.")
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
        help="Policy names to train and compare.",
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
    args = build_reference_args(reference_name)
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


def summarize_payload(payload, results_path: Path):
    heuristic_results = payload["evaluation"]["heuristics"]
    best_heuristic_cost = min(
        result["mean_cost"]
        for result in heuristic_results.values()
        if isinstance(result, dict) and "mean_cost" in result
    )
    learned_cost = payload["evaluation"]["learned_policy"]["mean_cost"]
    return {
        "reference": payload["experiment_name"].split("_soft_tree", 1)[0].split("_linear", 1)[0].split("_nn", 1)[0],
        "experiment_name": payload["experiment_name"],
        "policy_name": payload["policy_name"],
        "policy_architecture": payload["policy_architecture"],
        "action_adapter": payload["action_adapter"],
        "learned_mean_cost": learned_cost,
        "best_heuristic_cost": best_heuristic_cost,
        "gap_pct_vs_best_heuristic": 100.0 * (learned_cost / best_heuristic_cost - 1.0),
        "results_file": str(results_path),
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    summary = []
    for reference_name in parsed.references:
        for policy_name in parsed.policies:
            args = configure_args(parsed, reference_name, policy_name, root)
            payload, results_path = run_experiment(args)
            row = summarize_payload(payload, results_path)
            summary.append(row)
            print(
                f"{reference_name} {policy_name} "
                f"learned={row['learned_mean_cost']:.6f} "
                f"best_heur={row['best_heuristic_cost']:.6f} "
                f"gap={row['gap_pct_vs_best_heuristic']:.4f}%"
            )

    summary.sort(key=lambda row: (row["reference"], row["gap_pct_vs_best_heuristic"]))
    output_path = root / "screening_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
