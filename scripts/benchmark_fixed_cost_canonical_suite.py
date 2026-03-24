import argparse
import json
import sys
from copy import copy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.problems.lost_sales_fixed_order_cost.benchmark import benchmark_reference_instance
from invman.problems.lost_sales_fixed_order_cost.reference_instances import build_reference_args


COMMON_BUDGET = {
    "training_episodes": 5000,
    "es_population": 50,
    "horizon": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
}


EXPERIMENT_SPECS = [
    {
        "id": "linear_categorical_quantity",
        "policy_type": "linear",
        "policy_head": "categorical_quantity",
        "rollout_backend": "rust",
    },
    {
        "id": "linear_gated_ordinal_quantity",
        "policy_type": "linear",
        "policy_head": "gated_ordinal_quantity",
        "rollout_backend": "python",
    },
    {
        "id": "nn_categorical_quantity",
        "policy_type": "nn",
        "policy_head": "categorical_quantity",
        "rollout_backend": "rust",
        "hidden_dim": [50],
        "activation": "selu",
    },
    {
        "id": "nn_gated_ordinal_quantity",
        "policy_type": "nn",
        "policy_head": "gated_ordinal_quantity",
        "rollout_backend": "python",
        "hidden_dim": [50],
        "activation": "selu",
    },
    {
        "id": "soft_tree_depth2_linear_leaf",
        "policy_type": "soft_tree",
        "rollout_backend": "rust",
        "tree_depth": 2,
        "tree_temperature": 0.25,
        "tree_split_type": "oblique",
        "tree_leaf_type": "linear",
    },
    {
        "id": "soft_tree_depth1_linear_leaf",
        "policy_type": "soft_tree",
        "rollout_backend": "rust",
        "tree_depth": 1,
        "tree_temperature": 0.25,
        "tree_split_type": "oblique",
        "tree_leaf_type": "linear",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the canonical fixed-order-cost benchmark suite and render a paper-style summary table."
    )
    parser.add_argument("--reference", default="lit_pois_mu5_l4_p4_k5")
    parser.add_argument("--run_tag", default="fixed_cost_l4_canonical_suite_5k_paperlike")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--search_horizon", type=int, default=10000)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of experiment ids to run.",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse existing per-policy result JSONs when present instead of rerunning them.",
    )
    parser.add_argument(
        "--reuse_existing_summary",
        action="store_true",
        help="Reuse the existing suite summary heuristics block when present instead of recomputing it.",
    )
    return parser.parse_args()


def _suite_root(run_tag: str) -> Path:
    return PACKAGE_ROOT / "outputs" / "benchmarks" / run_tag


def _ensure_dirs(root: Path):
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)


def _configure_run_args(parsed, spec, root: Path):
    args = build_reference_args(parsed.reference)
    args.problem = "lost_sales_fixed_order_cost"
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = COMMON_BUDGET["training_episodes"]
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = COMMON_BUDGET["horizon"]
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
    args.policy_type = spec["policy_type"]
    args.rollout_backend = spec["rollout_backend"]
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    args.experiment_name = f"{parsed.run_tag}_{spec['id']}"

    if args.policy_type == "linear":
        args.policy_head = spec["policy_head"]
    elif args.policy_type == "nn":
        args.policy_head = spec["policy_head"]
        args.hidden_dim = spec["hidden_dim"]
        args.activation = spec["activation"]
    elif args.policy_type == "soft_tree":
        args.policy_head = "categorical_quantity"
        args.tree_depth = spec["tree_depth"]
        args.tree_temperature = spec["tree_temperature"]
        args.tree_split_type = spec["tree_split_type"]
        args.tree_leaf_type = spec["tree_leaf_type"]
    else:  # pragma: no cover
        raise NotImplementedError(spec["policy_type"])

    return args


def _render_markdown(summary):
    heuristic = summary["heuristics"]["evaluation"]
    best_heuristic_cost = min(
        heuristic[name]["mean_cost"] for name in ("s_s", "s_nq", "modified_s_s_q")
    )
    lines = [
        "# Canonical Fixed-Cost Benchmark Suite",
        "",
        f"Reference instance: `{summary['reference']}`",
        "",
        "## Heuristic Baseline",
        "",
        "| Policy | Params | Mean cost |",
        "| --- | --- | ---: |",
        f"| `s,S` | `{heuristic['s_s']['params']}` | `{heuristic['s_s']['mean_cost']:.5f}` |",
        f"| `s,nQ` | `{heuristic['s_nq']['params']}` | `{heuristic['s_nq']['mean_cost']:.5f}` |",
        f"| modified `s,S,q` | `{heuristic['modified_s_s_q']['params']}` | `{heuristic['modified_s_s_q']['mean_cost']:.5f}` |",
        "",
        "## Policy Function Approximators",
        "",
        "| Approximator | Architecture | Backend | Eval horizon | Mean cost | Gap vs best heuristic |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in summary["learned_policies"]:
        learned_cost = result["evaluation"]["learned_policy"]["mean_cost"]
        lines.append(
            "| {name} | `{arch}` | `{backend}` | `{horizon}` | `{cost:.5f}` | `{gap:.5f}` |".format(
                name=result["label"],
                arch=result["payload"]["policy_architecture"],
                backend=result["payload"]["rollout_backend"],
                horizon=result["payload"]["evaluation_horizon"],
                cost=learned_cost,
                gap=learned_cost - best_heuristic_cost,
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- training episodes: `{COMMON_BUDGET['training_episodes']}`",
            f"- ES population: `{COMMON_BUDGET['es_population']}`",
            f"- training horizon: `{COMMON_BUDGET['horizon']}`",
            f"- evaluation horizon: `{summary['eval_horizon']}`",
            f"- evaluation seeds: `{summary['eval_seeds']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _result_path_for(args) -> Path:
    return Path(args.results_dir) / f"{args.experiment_name}.json"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    result_path = _result_path_for(args)
    if reuse_existing and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return payload, result_path
    return run_experiment(args)


def _summary_paths(root: Path):
    return root / "fixed_cost_canonical_suite.json", root / "fixed_cost_canonical_suite.md"


def main():
    parsed = parse_args()
    root = _suite_root(parsed.run_tag)
    _ensure_dirs(root)
    selected_ids = set(parsed.only) if parsed.only else None
    summary_json, summary_md = _summary_paths(root)

    if parsed.reuse_existing_summary and summary_json.exists():
        existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        heuristic_summary = existing_summary["heuristics"]
    else:
        heuristic_summary = benchmark_reference_instance(
            parsed.reference,
            search_horizon=parsed.search_horizon,
            eval_horizon=parsed.eval_horizon,
            eval_seeds=parsed.eval_seeds,
            backend="rust",
            modified_search_mode="exhaustive",
        )

    learned_policy_results = []
    for spec in EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        args = _configure_run_args(parsed, spec, root)
        payload, result_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
        learned_policy_results.append(
            {
                "id": spec["id"],
                "label": spec["id"].replace("_", " "),
                "results_path": str(result_path),
                "payload": payload,
                "evaluation": payload["evaluation"],
            }
        )

    summary = {
        "reference": parsed.reference,
        "search_horizon": parsed.search_horizon,
        "eval_horizon": parsed.eval_horizon,
        "eval_seeds": parsed.eval_seeds,
        "heuristics": heuristic_summary,
        "learned_policies": learned_policy_results,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


if __name__ == "__main__":
    main()
