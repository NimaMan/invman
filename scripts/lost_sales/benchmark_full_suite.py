import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust

from invman.config import get_config
from invman.experiment_runner import run_experiment
from invman.policy_registry import apply_policy_name
from invman.utils import RunStatusTracker

# --- suite orchestration glue (grid + heuristic baselines from the Rust config) ---
# The 32-instance grid and per-instance heuristic reference costs live in the Rust
# crate (problems::lost_sales::reference_costs, via invman_rust); this script only
# carries the CMA-ES budget, policy roster, and arg-building/summary glue.
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


def _instance_params(ref):
    params = {
        "problem": "lost_sales",
        "demand_dist_name": ref["demand_kind"],
        "demand_rate": ref["demand_rate"],
        "lead_time": int(ref["lead_time"]),
        "holding_cost": ref["holding_cost"],
        "shortage_cost": ref["shortage_cost"],
        "max_order_size": 20,
        "horizon": COMMON_BUDGET["horizon_default"],
        "eval_horizon": COMMON_BUDGET["eval_horizon"],
        "eval_seeds": COMMON_BUDGET["eval_seeds"],
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
        "seed": 123,
    }
    if ref["demand_kind"] == "MarkovModulatedPoisson2":
        params.update(
            demand_lambda_low=ref["demand_lambda_low"],
            demand_lambda_high=ref["demand_lambda_high"],
            demand_p00=ref["demand_p00"],
            demand_p11=ref["demand_p11"],
        )
    return params


def build_reference_args(name):
    ref = _reference(name)
    args = get_config([])
    for key, value in _instance_params(ref).items():
        setattr(args, key, value)
    args.state_normalizer = "quantity_scale"
    args.state_scale = 20.0
    return args


def get_benchmark_grid(grid_name="xin2020_extended_lost_sales"):
    instances = []
    for name in invman_rust.lost_sales_reference_instance_names():
        if name == "vanilla_l4_p4_poisson5":
            continue
        ref = _reference(name)
        instances.append({
            "name": name,
            "description": (
                f"Lost-sales grid instance: {ref['demand_kind']} demand (mean 5), "
                f"lead time {ref['lead_time']}, shortage cost {int(ref['shortage_cost'])}, holding cost 1."
            ),
            "params": _instance_params(ref),
            "literature_metadata": {
                "benchmark_family": "Xin2020TechnicalModels",
                "reference_cost_source": ref["source"],
                "reported_values": dict(ref["costs"]),
            },
        })
    return {
        "name": grid_name,
        "grid_name": grid_name,
        "description": (
            "Vanilla lost-sales paper grid: {Poisson, Geometric, MMPP2+, MMPP2-} demand x "
            "shortage cost {4, 19} x lead time {4, 6, 8, 10}, mean demand 5, holding cost 1."
        ),
        "axes": {"lead_time": [4, 6, 8, 10], "shortage_cost": [4, 19],
                 "demand_case": ["poisson", "geometric", "mmpp2_pos", "mmpp2_neg"]},
        "num_instances": len(instances),
        "instances": instances,
    }


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


def resolved_protocol_budget(parsed):
    return {
        "training_episodes_default": int(parsed.training_episodes)
        if getattr(parsed, "training_episodes", None) is not None
        else COMMON_BUDGET["training_episodes_default"],
        "horizon_default": int(parsed.training_horizon)
        if getattr(parsed, "training_horizon", None) is not None
        else COMMON_BUDGET["horizon_default"],
    }


def configure_run_args(parsed, spec, root, reference_name, *, include_reference_in_experiment_name=True):
    args = build_reference_args(reference_name)
    budget = resolved_protocol_budget(parsed)
    args.problem = "lost_sales"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes_default"]
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = budget["horizon_default"]
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
        description="Run the full vanilla lost-sales paper benchmark suite over the configured instance grid."
    )
    parser.add_argument("--grid_name", default="xin2020_extended_lost_sales")
    parser.add_argument("--run_tag", default="lost_sales_full_grid_suite_paperlike")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument(
        "--training_episodes",
        type=int,
        default=None,
        help="Override CMA-ES training episodes for every instance.",
    )
    parser.add_argument(
        "--training_horizon",
        type=int,
        default=None,
        help="Override simulation horizon used during training for every instance.",
    )
    parser.add_argument("--state_scale", type=float, default=None)
    parser.add_argument("--references", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--reuse_existing_instance_summary", action="store_true")
    parser.add_argument(
        "--instance_jobs",
        type=int,
        default=1,
        help="Number of reference instances to process concurrently.",
    )
    parser.add_argument(
        "--skip_suite_summary",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _suite_root(run_tag: str) -> Path:
    return PACKAGE_ROOT / "outputs" / "benchmarks" / run_tag


def _ensure_dirs(root: Path):
    for dirname in ("results", "logs", "models", "instances"):
        (root / dirname).mkdir(parents=True, exist_ok=True)


def _summary_paths(root: Path):
    return root / "lost_sales_full_suite.json", root / "lost_sales_full_suite.md"


def _suite_status_path(root: Path):
    return root / "suite_status.json"


def _instance_summary_path(root: Path, reference_name: str) -> Path:
    return root / "instances" / f"{reference_name}.json"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    path = result_path_for(args)
    if reuse_existing and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    return run_experiment(args)


def _instance_protocol(parsed) -> dict:
    budget = resolved_protocol_budget(parsed)
    return {
        "training_episodes_default": budget["training_episodes_default"],
        "es_population": COMMON_BUDGET["es_population"],
        "training_horizon_default": budget["horizon_default"],
        "eval_horizon": parsed.eval_horizon,
        "eval_seeds": parsed.eval_seeds,
        "sigma_init": COMMON_BUDGET["sigma_init"],
        "warm_up_periods_ratio": 0.2,
    }


def _render_markdown(summary: dict) -> str:
    lines = [
        "# Vanilla Lost-Sales Full Grid Suite",
        "",
        f"Grid: `{summary['grid_name']}`",
        f"Instances: `{summary['num_instances']}`",
        "",
        "## Protocol",
        "",
        f"- training episodes: `{summary['protocol']['training_episodes_default']}`",
        f"- ES population: `{summary['protocol']['es_population']}`",
        f"- training horizon: `{summary['protocol']['training_horizon_default']}`",
        f"- evaluation horizon: `{summary['protocol']['eval_horizon']}`",
        f"- evaluation seeds: `{summary['protocol']['eval_seeds']}`",
        "",
        "## Aggregate Policy Summary",
        "",
        "| Policy | Mean relative gap vs best heuristic (%) | Better than best heuristic (count) | Better than literature CBS (count) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy_id, item in summary["aggregate"]["policies"].items():
        lines.append(
            f"| `{policy_id}` | `{item['mean_relative_gap_pct_vs_best_heuristic']:.4f}` | "
            f"`{item['better_than_best_heuristic_count']}/{summary['num_instances']}` | "
            f"`{item['better_than_literature_cbs_count']}/{item['instances_with_literature_cbs']}` |"
        )
    return "\n".join(lines) + "\n"


def _summarize_instances(instances: list[dict]) -> dict:
    policies = {}
    for spec in EXPERIMENT_SPECS:
        policy_id = spec["id"]
        costs = []
        rel_gaps = []
        better_count = 0
        better_than_cbs = 0
        instances_with_cbs = 0
        for instance in instances:
            if policy_id not in instance["learned_policies"]:
                continue
            learned_cost = instance["learned_policies"][policy_id]["evaluation"]["learned_policy"]["mean_cost"]
            best_heuristic = instance["comparative_summary"]["best_heuristic_cost"]
            costs.append(learned_cost)
            if best_heuristic is not None:
                rel_gaps.append(100.0 * (learned_cost - best_heuristic) / best_heuristic)
                if learned_cost < best_heuristic:
                    better_count += 1
            cbs = instance["literature_references"]["capped_base_stock"]["mean_cost"]
            if cbs is not None:
                instances_with_cbs += 1
                if learned_cost < cbs:
                    better_than_cbs += 1
        if not costs:
            continue
        policies[policy_id] = {
            "num_instances": len(costs),
            "mean_cost_across_instances": float(sum(costs) / len(costs)),
            "mean_relative_gap_pct_vs_best_heuristic": float(sum(rel_gaps) / len(rel_gaps)) if rel_gaps else None,
            "instances_with_best_heuristic": int(len(rel_gaps)),
            "better_than_best_heuristic_count": int(better_count),
            "better_than_literature_cbs_count": int(better_than_cbs),
            "instances_with_literature_cbs": int(instances_with_cbs),
        }
    return {"policies": policies}


def _build_instance_payload(parsed, root: Path, instance: dict, selected_ids: set[str] | None, tracker=None):
    reference_name = instance["name"]
    instance_summary_path = _instance_summary_path(root, reference_name)
    if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
        return json.loads(instance_summary_path.read_text(encoding="utf-8"))

    if tracker is not None:
        tracker.update("benchmarking_heuristics", reference_name=reference_name)
    heuristic_summary = benchmark_reference_instance(
        reference_name,
        eval_horizon=parsed.eval_horizon,
        eval_seeds=parsed.eval_seeds,
    )

    learned_policies = {}
    for spec in EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        if tracker is not None:
            tracker.update(
                "running_policy",
                reference_name=reference_name,
                policy_id=spec["id"],
            )
        args = configure_run_args(parsed, spec, root, reference_name)
        if parsed.state_scale is not None:
            args.state_scale = float(parsed.state_scale)
        payload, result_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
        learned_policies[spec["id"]] = {
            "results_path": str(result_path),
            "policy_spec": spec,
            "evaluation": payload["evaluation"],
            "payload": payload,
            "checkpoint_glob": str((root / "models" / f"{args.experiment_name}_*").resolve()),
        }

    heuristic_eval = heuristic_summary["evaluation"]
    available_heuristics = {
        name: entry for name, entry in heuristic_eval.items() if entry.get("mean_cost") is not None
    }
    if available_heuristics:
        best_heuristic_name, best_heuristic_entry = min(
            available_heuristics.items(),
            key=lambda kv: kv[1]["mean_cost"],
        )
        best_heuristic_cost = float(best_heuristic_entry["mean_cost"])
    else:
        best_heuristic_name, best_heuristic_cost = None, None
    comparative = {
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "policy_gaps": {},
    }
    for policy_id, item in learned_policies.items():
        learned_cost = float(item["evaluation"]["learned_policy"]["mean_cost"])
        if best_heuristic_cost is None:
            comparative["policy_gaps"][policy_id] = {
                "gap_vs_best_heuristic": None,
                "relative_gap_pct_vs_best_heuristic": None,
            }
        else:
            comparative["policy_gaps"][policy_id] = {
                "gap_vs_best_heuristic": learned_cost - best_heuristic_cost,
                "relative_gap_pct_vs_best_heuristic": 100.0 * (learned_cost - best_heuristic_cost) / best_heuristic_cost,
            }

    payload = {
        "reference_instance": reference_name,
        "reference_description": instance["description"],
        "params": instance["params"],
        "literature_metadata": instance["literature_metadata"],
        "protocol": _instance_protocol(parsed),
        "heuristics": heuristic_summary,
        "literature_references": {
            "optimal": heuristic_summary["optimal_reference"],
            "capped_base_stock": heuristic_summary["capped_base_stock_reference"],
        },
        "learned_policies": learned_policies,
        "comparative_summary": comparative,
    }
    instance_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _effective_instance_jobs(parsed, num_instances: int) -> int:
    requested = max(1, int(parsed.instance_jobs))
    total_rollout_workers = max(1, int(parsed.mp_num_processors))
    return max(1, min(requested, num_instances, total_rollout_workers))


def _shared_child_command_args(parsed, *, mp_num_processors: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--grid_name",
        parsed.grid_name,
        "--run_tag",
        parsed.run_tag,
        "--seed",
        str(parsed.seed),
        "--mp_num_processors",
        str(mp_num_processors),
        "--eval_horizon",
        str(parsed.eval_horizon),
        "--eval_seeds",
        str(parsed.eval_seeds),
        "--training_episodes",
        str(parsed.training_episodes) if getattr(parsed, "training_episodes", None) is not None else "",
        "--training_horizon",
        str(parsed.training_horizon) if getattr(parsed, "training_horizon", None) is not None else "",
        "--state_scale",
        str(parsed.state_scale) if parsed.state_scale is not None else "",
        "--instance_jobs",
        "1",
        "--skip_suite_summary",
    ]
    if command[command.index("--training_episodes") + 1] == "":
        del command[command.index("--training_episodes"):command.index("--training_episodes") + 2]
    if command[command.index("--training_horizon") + 1] == "":
        del command[command.index("--training_horizon"):command.index("--training_horizon") + 2]
    if command[command.index("--state_scale") + 1] == "":
        del command[command.index("--state_scale"):command.index("--state_scale") + 2]
    if parsed.same_seed:
        command.append("--same_seed")
    if parsed.reuse_existing:
        command.append("--reuse_existing")
    if parsed.reuse_existing_instance_summary:
        command.append("--reuse_existing_instance_summary")
    if parsed.only:
        command.extend(["--only", *parsed.only])
    return command


def _run_instance_subprocess(parsed, reference_name: str, *, mp_num_processors: int):
    command = _shared_child_command_args(parsed, mp_num_processors=mp_num_processors)
    command.extend(["--references", reference_name])
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)


def _collect_instance_payloads(parsed, root: Path, grid_instances: list[dict], selected_ids: set[str] | None, tracker=None):
    actual_jobs = _effective_instance_jobs(parsed, len(grid_instances))
    if actual_jobs == 1 or len(grid_instances) <= 1:
        return [
            _build_instance_payload(parsed, root, instance, selected_ids, tracker=tracker)
            for instance in grid_instances
        ]

    per_instance_mp_num_processors = max(1, int(parsed.mp_num_processors) // actual_jobs)
    pending_instances = []
    instance_payloads = []
    for instance in grid_instances:
        instance_summary_path = _instance_summary_path(root, instance["name"])
        if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
            instance_payloads.append(json.loads(instance_summary_path.read_text(encoding="utf-8")))
        else:
            pending_instances.append(instance)

    print(
        json.dumps(
            {
                "instance_parallelism": {
                    "instance_jobs": actual_jobs,
                    "per_instance_mp_num_processors": per_instance_mp_num_processors,
                    "pending_instances": [instance["name"] for instance in pending_instances],
                }
            },
            indent=2,
        )
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_jobs) as executor:
        futures = {
            executor.submit(
                _run_instance_subprocess,
                parsed,
                instance["name"],
                mp_num_processors=per_instance_mp_num_processors,
            ): instance["name"]
            for instance in pending_instances
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()

    for instance in pending_instances:
        instance_summary_path = _instance_summary_path(root, instance["name"])
        instance_payloads.append(json.loads(instance_summary_path.read_text(encoding="utf-8")))

    order = {instance["name"]: idx for idx, instance in enumerate(grid_instances)}
    instance_payloads.sort(key=lambda payload: order[payload["reference_instance"]])
    return instance_payloads


def main():
    parsed = parse_args()
    root = _suite_root(parsed.run_tag)
    _ensure_dirs(root)

    tracker = None
    tracker_cm = None
    if not parsed.skip_suite_summary:
        tracker_cm = RunStatusTracker(
            _suite_status_path(root),
            metadata={
                "run_tag": parsed.run_tag,
                "problem": "lost_sales",
                "grid_name": parsed.grid_name,
                "seed": int(parsed.seed),
            },
        )
        tracker = tracker_cm.__enter__()

    try:
        if tracker is not None:
            tracker.update("loading_grid")
        grid = get_benchmark_grid(parsed.grid_name)
        grid_instances = grid["instances"]
        if parsed.references is not None:
            requested = set(parsed.references)
            grid_instances = [instance for instance in grid_instances if instance["name"] in requested]
        if parsed.limit is not None:
            grid_instances = grid_instances[: int(parsed.limit)]

        selected_ids = set(parsed.only) if parsed.only else None
        if parsed.skip_suite_summary:
            instance_payloads = _collect_instance_payloads(
                parsed,
                root,
                grid_instances,
                selected_ids,
                tracker=None,
            )
        else:
            instance_payloads = _collect_instance_payloads(
                parsed,
                root,
                grid_instances,
                selected_ids,
                tracker=tracker,
            )

        summary = {
            "run_tag": parsed.run_tag,
            "grid_name": parsed.grid_name,
            "grid_description": grid["description"],
            "grid_axes": grid["axes"],
            "num_instances": len(instance_payloads),
            "protocol": _instance_protocol(parsed),
            "instances": instance_payloads,
            "aggregate": _summarize_instances(instance_payloads),
        }

        if parsed.skip_suite_summary:
            return

        if tracker is not None:
            tracker.update("writing_suite_summary", num_instances=len(instance_payloads))
        summary_json, summary_md = _summary_paths(root)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_md.write_text(_render_markdown(summary), encoding="utf-8")
        if tracker is not None:
            tracker.mark_completed(
                summary_json=str(summary_json),
                summary_md=str(summary_md),
                num_instances=len(instance_payloads),
            )
        print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))
    finally:
        if tracker_cm is not None:
            tracker_cm.__exit__(*sys.exc_info())


if __name__ == "__main__":
    main()
