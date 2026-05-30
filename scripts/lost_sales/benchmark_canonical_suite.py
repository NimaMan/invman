import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust

from invman.config import get_config
from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name

# --- suite orchestration glue (heuristic baselines from the Rust reference-cost config) ---
COMMON_BUDGET = {
    "training_episodes_default": 2000,
    "es_population": 64,
    "horizon_default": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
    "save_every": 1000,
}
EXPERIMENT_SPECS = [
    {"id": "linear_categorical_quantity_q20", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_sigmoid_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_soft_gated_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_direct_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "linear_hard_gated_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_soft_gated_ordinal_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_ordinal_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "soft_tree_depth1_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
    {"id": "soft_tree_depth2_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
]


def _reference(name):
    ref = invman_rust.lost_sales_reference_costs(name)
    if ref is None:
        raise KeyError(f"unknown lost-sales reference instance: {name}")
    return ref


def build_reference_args(name):
    ref = _reference(name)
    args = get_config([])
    args.problem = "lost_sales"
    args.demand_dist_name = ref["demand_kind"]
    args.demand_rate = ref["demand_rate"]
    args.lead_time = int(ref["lead_time"])
    args.holding_cost = ref["holding_cost"]
    args.shortage_cost = ref["shortage_cost"]
    args.max_order_size = 20
    args.track_demand = True
    args.warm_up_periods_ratio = 0.2
    args.state_normalizer = "quantity_scale"
    args.state_scale = 20.0
    if ref["demand_kind"] == "MarkovModulatedPoisson2":
        args.demand_lambda_low = ref["demand_lambda_low"]
        args.demand_lambda_high = ref["demand_lambda_high"]
        args.demand_p00 = ref["demand_p00"]
        args.demand_p11 = ref["demand_p11"]
    return args


def _cost_summary(value):
    return {"mean_cost": None if value is None else float(value),
            "available": value is not None, "source": "reference_config"}


def benchmark_reference_instance(name, *, eval_horizon=None, eval_seeds=None, **_ignored):
    costs = _reference(name)["costs"]
    return {
        "reference_instance": name,
        "evaluation": {
            "myopic1": _cost_summary(costs["myopic1"]),
            "myopic2": _cost_summary(costs["myopic2"]),
            "svbs": _cost_summary(costs["svbs"]),
        },
        "optimal_reference": _cost_summary(costs["optimal"]),
        "capped_base_stock_reference": _cost_summary(costs["capped_base_stock"]),
    }


def configure_run_args(parsed, spec, root, reference_name, *, include_reference_in_experiment_name=True):
    args = build_reference_args(reference_name)
    args.problem = "lost_sales"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = int(getattr(parsed, "training_episodes", None) or COMMON_BUDGET["training_episodes_default"])
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = int(getattr(parsed, "training_horizon", None) or COMMON_BUDGET["horizon_default"])
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
    args.save_every = COMMON_BUDGET["save_every"]
    args.max_order_size = 20
    args.policy_name = spec["id"]
    apply_policy_name(args)
    args.rollout_backend = spec["rollout_backend"]
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    if include_reference_in_experiment_name:
        args.experiment_name = f"{parsed.run_tag}_{reference_name}_{spec['id']}"
    else:
        args.experiment_name = f"{parsed.run_tag}_{spec['id']}"
    return args


def result_path_for(args):
    return Path(args.results_dir) / f"{args.experiment_name}.json"
# --- end suite orchestration glue ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the canonical vanilla lost-sales benchmark suite and render a paper-style summary table."
    )
    parser.add_argument("--reference", default="vanilla_l4_p4_poisson5")
    parser.add_argument("--run_tag", default="lost_sales_l4_canonical_suite_paperlike")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--reuse_existing_summary", action="store_true")
    return parser.parse_args()


def _suite_root(run_tag: str) -> Path:
    return PACKAGE_ROOT / "outputs" / "benchmarks" / run_tag


def _ensure_dirs(root: Path):
    for dirname in ("results", "logs", "models"):
        (root / dirname).mkdir(parents=True, exist_ok=True)


def _summary_paths(root: Path):
    return root / "lost_sales_canonical_suite.json", root / "lost_sales_canonical_suite.md"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    result_path = result_path_for(args)
    if reuse_existing and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return payload, result_path
    return run_experiment(args)


def _render_markdown(summary):
    heuristic = summary["heuristics"]["evaluation"]
    heuristic_costs = [
        heuristic[name]["mean_cost"]
        for name in ("myopic1", "myopic2", "svbs")
        if heuristic[name]["mean_cost"] is not None
    ]
    best_heuristic_cost = min(heuristic_costs) if heuristic_costs else None
    lines = [
        "# Canonical Vanilla Lost-Sales Benchmark Suite",
        "",
        f"Reference instance: `{summary['reference']}`",
        "",
        "## Literature Anchors",
        "",
        f"- optimal: `{summary['heuristics']['optimal_reference']['mean_cost']}`",
        f"- capped base-stock: `{summary['heuristics']['capped_base_stock_reference']['mean_cost']}`",
        "",
        "## Heuristic Baseline",
        "",
        "| Policy | Mean cost | Max order observed |",
        "| --- | ---: | ---: |",
        f"| `myopic1` | `{heuristic['myopic1']['mean_cost']:.5f}` | `{heuristic['myopic1']['max_order_observed']}` |",
        f"| `myopic2` | `{heuristic['myopic2']['mean_cost']:.5f}` | `{heuristic['myopic2']['max_order_observed']}` |",
        f"| `svbs` | `{heuristic['svbs']['mean_cost']:.5f}` | `{heuristic['svbs']['max_order_observed']}` |",
        "",
        "## Policy Function Approximators",
        "",
        "| Approximator | Architecture | qbar | Backend | Mean cost | Gap vs best heuristic |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for result in summary["learned_policies"]:
        learned_cost = result["evaluation"]["learned_policy"]["mean_cost"]
        lines.append(
            "| {name} | `{arch}` | `{qbar}` | `{backend}` | `{cost:.5f}` | `{gap:.5f}` |".format(
                name=result["label"],
                arch=result["payload"]["policy_architecture"],
                qbar=result["payload"]["max_order_size"],
                backend=result["payload"]["rollout_backend"],
                cost=learned_cost,
                gap=learned_cost - best_heuristic_cost,
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- training episodes: `{COMMON_BUDGET['training_episodes_default']}`",
            f"- ES population: `{COMMON_BUDGET['es_population']}`",
            f"- training horizon: `{COMMON_BUDGET['horizon_default']}`",
            f"- evaluation horizon: `{summary['eval_horizon']}`",
            f"- evaluation seeds: `{summary['eval_seeds']}`",
        ]
    )
    return "\n".join(lines) + "\n"


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
            eval_horizon=parsed.eval_horizon,
            eval_seeds=parsed.eval_seeds,
        )

    learned_policy_results = []
    for spec in EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        args = configure_run_args(
            parsed,
            spec,
            root,
            parsed.reference,
            include_reference_in_experiment_name=False,
        )
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
