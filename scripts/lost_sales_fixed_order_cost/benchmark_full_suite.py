import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.problems.lost_sales_fixed_order_cost.benchmark import benchmark_reference_instance
from invman.problems.lost_sales_fixed_order_cost.experiment_spec import (
    COMMON_BUDGET,
    EXPERIMENT_SPECS,
    configure_run_args,
    result_path_for,
)
from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    get_benchmark_grid,
    get_reference_instance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full fixed-cost literature-aligned benchmark suite over the configured instance grid."
    )
    parser.add_argument("--grid_name", default="literature_subset_poisson_mu5")
    parser.add_argument("--run_tag", default="fixed_cost_full_grid_suite_5k_paperlike")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--search_horizon", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument(
        "--references",
        nargs="+",
        default=None,
        help="Optional explicit list of reference instance names to run.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of experiment ids to run.",
    )
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
    return root / "fixed_cost_full_suite.json", root / "fixed_cost_full_suite.md"


def _instance_summary_path(root: Path, reference_name: str) -> Path:
    return root / "instances" / f"{reference_name}.json"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    path = result_path_for(args)
    if reuse_existing and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    return run_experiment(args)


def _optimal_reference(reference_name: str) -> dict:
    reference = get_reference_instance(reference_name)
    published = reference.get("benchmark_anchors", {}).get("published_optimal_reference")
    if published is not None:
        return {
            "available": bool(published.get("available", True)),
            "reference_instance": reference_name,
            "source": "Bijvank2015ParametricPolicies",
            "mean_cost": float(published["mean_cost"]),
        }
    return {
        "available": False,
        "reference_instance": reference_name,
        "source": "Bijvank2015ParametricPolicies",
        "note": "No exact per-instance optimum is currently encoded for this literature-aligned subset.",
    }


def _instance_protocol(parsed) -> dict:
    return {
        "training_episodes": COMMON_BUDGET["training_episodes"],
        "es_population": COMMON_BUDGET["es_population"],
        "training_horizon": COMMON_BUDGET["horizon"],
        "eval_horizon": parsed.eval_horizon,
        "eval_seeds": parsed.eval_seeds,
        "sigma_init": COMMON_BUDGET["sigma_init"],
        "warm_up_periods_ratio": 0.2,
    }


def _render_markdown(summary: dict) -> str:
    lines = [
        "# Fixed-Cost Full Grid Suite",
        "",
        f"Grid: `{summary['grid_name']}`",
        f"Instances: `{summary['num_instances']}`",
        "",
        "## Protocol",
        "",
        f"- training episodes: `{summary['protocol']['training_episodes']}`",
        f"- ES population: `{summary['protocol']['es_population']}`",
        f"- training horizon: `{summary['protocol']['training_horizon']}`",
        f"- evaluation horizon: `{summary['protocol']['eval_horizon']}`",
        f"- evaluation seeds: `{summary['protocol']['eval_seeds']}`",
        "",
        "## Aggregate Policy Summary",
        "",
        "| Policy | Mean relative gap vs best heuristic (%) | Better than best heuristic (count) | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for policy_id, item in summary["aggregate"]["policies"].items():
        lines.append(
            f"| `{policy_id}` | `{item['mean_relative_gap_pct_vs_best_heuristic']:.4f}` | "
            f"`{item['better_than_best_heuristic_count']}/{summary['num_instances']}` | `{item['status']}` |"
        )

    lines.extend(
        [
            "",
            "## Instance Files",
            "",
            "Each per-instance JSON contains:",
            "",
            "- reference metadata and literature tags",
            "- heuristic search configuration and winning parameters",
            "- heuristic long-run evaluations",
            "- optimal-reference availability metadata",
            "- learned-policy evaluations and result file paths",
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize_instances(instances: list[dict]) -> dict:
    policy_ids = [spec["id"] for spec in EXPERIMENT_SPECS]
    policies = {}
    for policy_id in policy_ids:
        costs = []
        rel_gaps = []
        better_count = 0
        status = next(spec.get("status", "trusted") for spec in EXPERIMENT_SPECS if spec["id"] == policy_id)
        for instance in instances:
            if policy_id not in instance["learned_policies"]:
                continue
            learned_cost = instance["learned_policies"][policy_id]["evaluation"]["learned_policy"]["mean_cost"]
            best_heuristic = instance["comparative_summary"]["best_heuristic_cost"]
            costs.append(learned_cost)
            rel_gap = 100.0 * (learned_cost - best_heuristic) / best_heuristic
            rel_gaps.append(rel_gap)
            if learned_cost < best_heuristic:
                better_count += 1
        if not costs:
            continue
        policies[policy_id] = {
            "num_instances": len(costs),
            "mean_cost_across_instances": float(sum(costs) / len(costs)),
            "mean_relative_gap_pct_vs_best_heuristic": float(sum(rel_gaps) / len(rel_gaps)),
            "better_than_best_heuristic_count": int(better_count),
            "status": status,
        }

    return {"policies": policies}


def _build_instance_payload(parsed, root: Path, instance: dict, selected_ids: set[str] | None):
    reference_name = instance["name"]
    instance_summary_path = _instance_summary_path(root, reference_name)
    if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
        return json.loads(instance_summary_path.read_text(encoding="utf-8"))

    heuristic_summary = benchmark_reference_instance(
        reference_name,
        search_horizon=parsed.search_horizon,
        eval_horizon=parsed.eval_horizon,
        eval_seeds=parsed.eval_seeds,
        backend="rust",
        modified_search_mode="exhaustive",
    )

    learned_policies = {}
    for spec in EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        args = configure_run_args(parsed, spec, root, reference_name)
        payload, result_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
        learned_policies[spec["id"]] = {
            "results_path": str(result_path),
            "policy_spec": spec,
            "evaluation": payload["evaluation"],
            "payload": payload,
            "checkpoint_glob": str((root / "models" / f"{args.experiment_name}_*").resolve()),
        }

    heuristic_eval = heuristic_summary["evaluation"]
    best_heuristic_name, best_heuristic_entry = min(
        heuristic_eval.items(),
        key=lambda kv: kv[1]["mean_cost"],
    )
    best_heuristic_cost = float(best_heuristic_entry["mean_cost"])

    comparative = {
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "policy_gaps": {},
    }
    for policy_id, item in learned_policies.items():
        learned_cost = float(item["evaluation"]["learned_policy"]["mean_cost"])
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
        "optimal_reference": _optimal_reference(reference_name),
        "heuristics": heuristic_summary,
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
        "--instance_jobs",
        "1",
        "--skip_suite_summary",
    ]
    if parsed.same_seed:
        command.append("--same_seed")
    if parsed.search_horizon is not None:
        command.extend(["--search_horizon", str(parsed.search_horizon)])
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


def _collect_instance_payloads(parsed, root: Path, grid_instances: list[dict], selected_ids: set[str] | None):
    actual_jobs = _effective_instance_jobs(parsed, len(grid_instances))
    if actual_jobs == 1 or len(grid_instances) <= 1:
        return [
            _build_instance_payload(parsed, root, instance, selected_ids)
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

    grid = get_benchmark_grid(parsed.grid_name)
    grid_instances = grid["instances"]
    if parsed.references is not None:
        requested = set(parsed.references)
        grid_instances = [instance for instance in grid_instances if instance["name"] in requested]
    if parsed.limit is not None:
        grid_instances = grid_instances[: int(parsed.limit)]

    selected_ids = set(parsed.only) if parsed.only else None

    instance_payloads = _collect_instance_payloads(parsed, root, grid_instances, selected_ids)

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

    summary_json, summary_md = _summary_paths(root)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


if __name__ == "__main__":
    main()
