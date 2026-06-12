import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name, get_policy_spec, make_soft_tree_policy_name
from scripts.dual_sourcing.dual_sourcing_benchmark_lib import (
    build_reference_args,
    evaluate_default_heuristics,
)


DEFAULT_REFERENCES = [
    "dual_l2_ce105",
    "dual_l2_ce110",
    "dual_l3_ce105",
    "dual_l3_ce110",
    "dual_l4_ce105",
    "dual_l4_ce110",
]

DEFAULT_POLICIES = [
    {
        "id": "tree_base_surge",
        "label": "Soft tree, base-surge targets",
        "policy_name": "soft_tree_base_surge_targets",
        "structure_family": "soft_tree",
        "backbone": "tree_oblique_linear",
        "control_family": "base_surge",
    },
    {
        "id": "tree_capped_dual_index",
        "label": "Soft tree, capped dual-index targets",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="capped_dual_index_targets",
        ),
        "structure_family": "soft_tree",
        "backbone": "tree_oblique_linear",
        "control_family": "capped_dual_index",
    },
    {
        "id": "tree_delta",
        "label": "Soft tree, dual-index delta targets",
        "policy_name": "soft_tree_dual_index_delta_targets",
        "structure_family": "soft_tree",
        "backbone": "tree_oblique_linear",
        "control_family": "delta",
    },
    {
        "id": "tree_capped_delta",
        "label": "Soft tree, capped dual-index delta",
        "policy_name": "soft_tree_capped_dual_index_delta_targets",
        "structure_family": "soft_tree",
        "backbone": "tree_oblique_linear",
        "control_family": "capped_delta",
    },
    {
        "id": "tree_smallcap_delta",
        "label": "Soft tree, small-cap capped dual-index delta",
        "policy_name": "soft_tree_capped_dual_index_delta_smallcap_targets",
        "structure_family": "soft_tree",
        "backbone": "tree_oblique_linear",
        "control_family": "smallcap_delta",
    },
    {
        "id": "tree_axis_constant_smallcap_delta",
        "label": "Soft tree, axis-constant small-cap capped dual-index delta",
        "policy_name": "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
        "structure_family": "soft_tree",
        "backbone": "tree_axis_constant",
        "control_family": "smallcap_delta",
    },
]

BUDGETS = {
    "screening": {
        "training_episodes": 300,
        "es_population": 8,
        "es_population_sampling": "fixed",
        "horizon": 1000,
        "eval_horizon": 5000,
        "eval_seeds": 2,
        "sigma_init": 3.0,
    },
    "promotion": {
        "training_episodes": 800,
        "es_population": 16,
        "es_population_sampling": "fixed",
        "horizon": 1500,
        "eval_horizon": 10000,
        "eval_seeds": 3,
        "sigma_init": 3.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a six-row dual-sourcing factor screen over a curated set of policy families."
    )
    parser.add_argument("--run_tag", default="dual_sourcing_factor_screen_v1")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    parser.add_argument("--references", nargs="+", default=DEFAULT_REFERENCES)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of policy ids from DEFAULT_POLICIES.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def _selected_policies(only_ids: list[str] | None):
    if only_ids is None:
        return DEFAULT_POLICIES
    by_id = {item["id"]: item for item in DEFAULT_POLICIES}
    missing = [policy_id for policy_id in only_ids if policy_id not in by_id]
    if missing:
        known = ", ".join(sorted(by_id))
        raise ValueError(
            f"Unknown factor-screen policy ids {missing}. Available ids: {known}"
        )
    return [by_id[policy_id] for policy_id in only_ids]


def _configure_args(parsed, reference_name: str, policy_item: dict, root: Path):
    budget = BUDGETS[parsed.budget]
    args = build_reference_args(reference_name)
    args.problem = "dual_sourcing"
    args.reference_instance = reference_name
    args.policy_name = policy_item["policy_name"]
    apply_policy_name(args)
    args.rollout_backend = "rust"
    args.training_method = "cma"
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.es_population_sampling = budget["es_population_sampling"]
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]
    args.sigma_init = float(budget["sigma_init"])
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{reference_name}_{policy_item['id']}"
    return args


def _result_path(args) -> Path:
    return Path(args.results_dir) / f"{args.experiment_name}.json"


def _load_or_run(args, reuse_existing: bool):
    path = _result_path(args)
    if reuse_existing and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    return run_experiment(args)


def _summarize_row(reference_name: str, policy_item: dict, payload: dict, result_path: Path):
    reference_args = build_reference_args(reference_name)
    heuristic_results = payload["evaluation"].get("heuristics") or evaluate_default_heuristics(
        reference_args,
        seed=int(getattr(reference_args, "seed", 123)),
        horizon=int(getattr(reference_args, "horizon", 6000)),
    )
    best_heuristic_name, best_heuristic = min(
        (
            (name, summary)
            for name, summary in heuristic_results.items()
            if isinstance(summary, dict) and "mean_cost" in summary
        ),
        key=lambda item: float(item[1]["mean_cost"]),
    )
    learned = payload["evaluation"]["learned_policy"]
    policy_spec = get_policy_spec(SimpleNamespace(policy_name=policy_item["policy_name"]))
    return {
        "reference": reference_name,
        "policy_id": policy_item["id"],
        "label": policy_item["label"],
        "policy_name": policy_item["policy_name"],
        "structure_family": policy_item["structure_family"],
        "backbone": policy_item["backbone"],
        "control_family": policy_item["control_family"],
        "policy_architecture": payload["policy_architecture"],
        "tree_depth": policy_spec.tree_depth,
        "tree_split_type": policy_spec.tree_split_type,
        "tree_leaf_type": policy_spec.tree_leaf_type,
        "learned_mean_cost": float(learned["mean_cost"]),
        "learned_std_cost": float(learned["std_cost"]),
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": float(best_heuristic["mean_cost"]),
        "gap_vs_best_heuristic": float(learned["mean_cost"]) - float(best_heuristic["mean_cost"]),
        "gap_pct_vs_best_heuristic": 100.0
        * (float(learned["mean_cost"]) / float(best_heuristic["mean_cost"]) - 1.0),
        "results_file": str(result_path),
    }


def _aggregate(rows: list[dict]):
    by_policy = {}
    by_reference = {}
    for row in rows:
        by_policy.setdefault(row["policy_id"], []).append(row)
        by_reference.setdefault(row["reference"], []).append(row)

    aggregate_policies = {}
    for policy_id, items in sorted(by_policy.items()):
        sorted_gaps = sorted(row["gap_pct_vs_best_heuristic"] for row in items)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        mean_gap = sum(sorted_gaps) / len(sorted_gaps)
        wins = sum(1 for row in items if row["gap_vs_best_heuristic"] < 0.0)
        aggregate_policies[policy_id] = {
            "label": items[0]["label"],
            "structure_family": items[0]["structure_family"],
            "backbone": items[0]["backbone"],
            "control_family": items[0]["control_family"],
            "mean_gap_pct_vs_best_heuristic": mean_gap,
            "median_gap_pct_vs_best_heuristic": median_gap,
            "wins_vs_best_heuristic": wins,
            "num_instances": len(items),
        }

    row_summary = []
    for reference, items in sorted(by_reference.items()):
        best_item = min(items, key=lambda row: row["gap_pct_vs_best_heuristic"])
        row_summary.append(
            {
                "reference": reference,
                "best_policy_id": best_item["policy_id"],
                "best_gap_pct_vs_best_heuristic": best_item["gap_pct_vs_best_heuristic"],
            }
        )

    return {"policy_summary": aggregate_policies, "row_summary": row_summary}


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    selected_policies = _selected_policies(parsed.only)
    for reference_name in parsed.references:
        for policy_item in selected_policies:
            args = _configure_args(parsed, reference_name, policy_item, root)
            payload, result_path = _load_or_run(args, reuse_existing=parsed.reuse_existing)
            row = _summarize_row(reference_name, policy_item, payload, result_path)
            rows.append(row)
            print(
                f"{reference_name} {policy_item['id']} "
                f"gap={row['gap_pct_vs_best_heuristic']:.4f}% "
                f"learned={row['learned_mean_cost']:.6f} "
                f"best_heur={row['best_heuristic_cost']:.6f}"
            )

    rows.sort(key=lambda row: (row["reference"], row["gap_pct_vs_best_heuristic"]))
    summary = {
        "run_tag": parsed.run_tag,
        "budget": parsed.budget,
        "references": list(parsed.references),
        "policies": selected_policies,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    out_path = root / "factor_screen_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
