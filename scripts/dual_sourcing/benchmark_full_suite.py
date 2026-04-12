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
from invman.problems.dual_sourcing.benchmark import evaluate_default_heuristics
from invman.problems.dual_sourcing.experiment_spec import (
    COMMON_BUDGET,
    DEFAULT_BUDGET,
    EXPERIMENT_SPECS,
    configure_run_args,
    get_budget_config,
    result_path_for,
)
from invman.problems.dual_sourcing.reference_instances import (
    GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME,
    build_grid_instances,
    get_benchmark_grid,
)
from invman.utils import RunStatusTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the dual-sourcing literature-aligned benchmark suite over the Gijs Figure 9 instance family."
    )
    parser.add_argument("--grid_name", default=GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME)
    parser.add_argument("--run_tag", default="dual_sourcing_gijs_structured_screening")
    parser.add_argument("--budget", choices=sorted(COMMON_BUDGET), default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--eval_seeds", type=int, default=None)
    parser.add_argument("--references", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Optional subset of experiment ids to run.")
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


def _budget(parsed):
    return get_budget_config(parsed.budget)


def _effective_eval_horizon(parsed) -> int:
    return int(parsed.eval_horizon if parsed.eval_horizon is not None else _budget(parsed)["eval_horizon"])


def _effective_eval_seeds(parsed) -> int:
    return int(parsed.eval_seeds if parsed.eval_seeds is not None else _budget(parsed)["eval_seeds"])


def _suite_root(run_tag: str) -> Path:
    return PACKAGE_ROOT / "outputs" / "benchmarks" / run_tag


def _ensure_dirs(root: Path):
    for dirname in ("results", "logs", "models", "instances"):
        (root / dirname).mkdir(parents=True, exist_ok=True)


def _summary_paths(root: Path):
    return root / "dual_sourcing_full_suite.json", root / "dual_sourcing_full_suite.md"


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
    budget = _budget(parsed)
    return {
        "budget": parsed.budget,
        "training_episodes": budget["training_episodes"],
        "es_population": budget["es_population"],
        "es_population_sampling": budget.get("es_population_sampling", "fixed"),
        "es_population_candidates": budget.get("es_population_candidates"),
        "es_population_probabilities": budget.get("es_population_probabilities"),
        "training_horizon": budget["horizon"],
        "eval_horizon": _effective_eval_horizon(parsed),
        "eval_seeds": _effective_eval_seeds(parsed),
        "sigma_init": budget["sigma_init"],
    }


def _render_markdown(summary: dict) -> str:
    lines = [
        "# Dual-Sourcing Full Grid Suite",
        "",
        f"Grid: `{summary['grid_name']}`",
        f"Instances: `{summary['num_instances']}`",
        f"Budget: `{summary['protocol']['budget']}`",
        "",
        "## Protocol",
        "",
        f"- training episodes: `{summary['protocol']['training_episodes']}`",
        f"- ES population: `{summary['protocol']['es_population']}`",
        f"- ES population sampling: `{summary['protocol']['es_population_sampling']}`",
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
            "- benchmark parameters and literature metadata",
            "- heuristic evaluations on the Gijs Figure 9 instance family",
            "- published Figure 9 optimality-gap labels carried as literature metadata",
            "- learned-policy evaluations and relative gaps vs the best heuristic",
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize_instances(instances: list[dict]) -> dict:
    policies = {}
    for spec in EXPERIMENT_SPECS:
        policy_id = spec["id"]
        costs = []
        rel_gaps = []
        better_count = 0
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
            "label": spec["label"],
            "status": spec.get("status", "candidate"),
            "num_instances": len(costs),
            "mean_cost_across_instances": float(sum(costs) / len(costs)),
            "mean_relative_gap_pct_vs_best_heuristic": float(sum(rel_gaps) / len(rel_gaps)),
            "better_than_best_heuristic_count": int(better_count),
        }
    return {"policies": policies}


def _heuristic_gap_check(instance: dict) -> dict:
    return {
        "published_optimality_gap_pct": instance["literature_metadata"].get("published_optimality_gap_pct", {}),
        "repo_optimality_gap_pct": None,
        "repo_gap_minus_paper_pct": None,
        "note": (
            "The full-grid training suite skips bounded-DP reproduction on the heavier l_r=3,4 rows. "
            "Use scripts/dual_sourcing/validate_reference_grid.py or the Rust verification path "
            "for explicit literature-gap reproduction against Figure 9."
        ),
    }


def _build_instance_payload(parsed, root: Path, instance: dict, selected_ids: set[str] | None, tracker=None):
    reference_name = instance["name"]
    instance_summary_path = _instance_summary_path(root, reference_name)
    if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
        return json.loads(instance_summary_path.read_text(encoding="utf-8"))

    if tracker is not None:
        tracker.update("benchmarking_heuristics", reference_name=reference_name)

    benchmark_args = configure_run_args(
        parsed,
        {
            "id": "__benchmark_probe__",
            "policy_name": EXPERIMENT_SPECS[0]["policy_name"],
            "rollout_backend": "rust",
        },
        root,
        reference_name,
    )
    benchmark_args.horizon = int(instance["search"]["search_horizon"])
    benchmark_args.eval_horizon = _effective_eval_horizon(parsed)
    benchmark_args.eval_seeds = _effective_eval_seeds(parsed)
    benchmark_args.seed = int(instance["search"]["search_seed"])

    heuristic_summary = {
        "heuristics": evaluate_default_heuristics(benchmark_args),
        "bounded_dp": None,
    }

    learned_policies = {}
    for spec in EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        if tracker is not None:
            tracker.update("running_policy", reference_name=reference_name, policy_id=spec["id"])
        args = configure_run_args(parsed, spec, root, reference_name)
        payload, result_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
        learned_policies[spec["id"]] = {
            "results_path": str(result_path),
            "policy_spec": spec,
            "evaluation": payload["evaluation"],
            "payload": payload,
            "checkpoint_glob": str((root / "models" / f"{args.experiment_name}_*").resolve()),
        }

    heuristic_eval = heuristic_summary["heuristics"]
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
        "search": instance["search"],
        "evaluation": instance["evaluation"],
        "literature_metadata": instance["literature_metadata"],
        "protocol": _instance_protocol(parsed),
        "heuristics": heuristic_summary,
        "literature_gap_check": _heuristic_gap_check(instance),
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
        "--budget",
        parsed.budget,
        "--seed",
        str(parsed.seed),
        "--mp_num_processors",
        str(mp_num_processors),
        "--instance_jobs",
        "1",
        "--skip_suite_summary",
    ]
    if parsed.eval_horizon is not None:
        command.extend(["--eval_horizon", str(parsed.eval_horizon)])
    if parsed.eval_seeds is not None:
        command.extend(["--eval_seeds", str(parsed.eval_seeds)])
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
                "problem": "dual_sourcing",
                "grid_name": parsed.grid_name,
                "seed": int(parsed.seed),
                "budget": parsed.budget,
            },
        )
        tracker = tracker_cm.__enter__()

    try:
        if tracker is not None:
            tracker.update("loading_grid")
        grid = get_benchmark_grid(parsed.grid_name)
        grid_instances = build_grid_instances(parsed.grid_name)
        if parsed.references is not None:
            requested = set(parsed.references)
            grid_instances = [instance for instance in grid_instances if instance["name"] in requested]
        if parsed.limit is not None:
            grid_instances = grid_instances[: int(parsed.limit)]

        selected_ids = set(parsed.only) if parsed.only else None
        instance_payloads = _collect_instance_payloads(parsed, root, grid_instances, selected_ids, tracker=tracker)

        summary = {
            "run_tag": parsed.run_tag,
            "grid_name": parsed.grid_name,
            "grid_description": grid["description"],
            "grid_axes": {
                "regular_lead_time": list(grid["regular_lead_times"]),
                "expedited_order_cost": list(grid["expedited_order_costs"]),
            },
            "num_instances": len(instance_payloads),
            "protocol": _instance_protocol(parsed),
            "selected_policies": [
                spec["id"]
                for spec in EXPERIMENT_SPECS
                if selected_ids is None or spec["id"] in selected_ids
            ],
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
