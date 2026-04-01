import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policies.registry import apply_policy_name, make_dense_policy_name
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
        "training_episodes": 800,
        "es_population": 8,
        "horizon": 2000,
        "eval_horizon": 10000,
        "eval_seeds": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare dual-sourcing policy backbones under a fixed training budget.")
    parser.add_argument("--run_tag", default="dual_sourcing_backbones", help="Namespace used for outputs.")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening", help="Fixed experiment budget.")
    parser.add_argument("--reference", default="dual_l4_ce110", help="Named dual-sourcing reference instance.")
    parser.add_argument("--policy_backbones", nargs="+", choices=["linear", "nn"], default=["linear", "nn"])
    parser.add_argument(
        "--action_adapters",
        nargs="+",
        default=["identity", "base_surge_targets"],
        help="Structured action adapters to compare for linear/nn policies.",
    )
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[16, 16], help="Hidden layers for the neural policy.")
    parser.add_argument("--activation", choices=["selu", "gelu", "relu"], default="selu")
    parser.add_argument("--sigma_init", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within a CMA batch.")
    return parser.parse_args()


def _prepare_args(parsed, root, policy_backbone, action_adapter):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(parsed.reference)
    args.problem = "dual_sourcing"
    args.policy_name = make_dense_policy_name(
        policy_backbone,
        "bounded_quantity",
        hidden_dim=parsed.hidden_dim if policy_backbone == "nn" else None,
        activation=parsed.activation if policy_backbone == "nn" else None,
        action_adapter=action_adapter,
    )
    apply_policy_name(args)
    args.rollout_backend = "python"
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
    heuristic_cost = min(
        summary["mean_cost"]
        for summary in payload["evaluation"]["heuristics"].values()
        if isinstance(summary, dict) and "mean_cost" in summary
    )
    return {
        "experiment_name": payload["experiment_name"],
        "policy_name": payload["policy_name"],
        "policy_backbone": payload["policy_backbone"],
        "policy_architecture": payload["policy_architecture"],
        "action_adapter": payload.get("action_adapter", "identity"),
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
    for policy_backbone in parsed.policy_backbones:
        for action_adapter in parsed.action_adapters:
            args = _prepare_args(parsed, root, policy_backbone, action_adapter)
            payload, results_path = run_experiment(args)
            payload["results_file"] = str(results_path)
            results.append(_summarize_result(payload))

    results.sort(key=lambda item: item["learned_mean_cost"])
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "reference": parsed.reference,
        "policy_backbones": parsed.policy_backbones,
        "action_adapters": parsed.action_adapters,
        "results": results,
        "best_result": results[0] if results else None,
    }

    summary_path = root / f"dual_sourcing_backbones_{parsed.budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
