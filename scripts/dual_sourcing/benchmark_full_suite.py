"""
Dual-sourcing full benchmark suite over the Gijsbrechts-2022 Figure-9 family.

OBJECTIVE
    Train the chosen soft-tree CMA-ES policy spec(s) across the six published
    dual-sourcing rows and write per-instance summaries comparing the learned
    policy cost to the four Gijsbrechts heuristics (single / dual / capped-dual
    index, tailored base-surge), the optional bounded-DP optimum, and the
    published Figure-9 optimality-gap labels (including the A3C DRL baseline).
    Dual sourcing routes through Rust and is soft_tree-ONLY, so the learned policy
    roster (EXPERIMENT_SPECS in dual_sourcing_benchmark_lib) is soft-tree
    structures over the capped-dual-index / dual-index control bases.

WHY (requirements -> objective)
    * Grid + reference + heuristic + optimal data all come from invman_rust via
      dual_sourcing_benchmark_lib (the Python problem package was deleted). The
      experiment payload's heuristics block is EMPTY for dual sourcing, so the
      suite computes heuristics itself with the Rust search bindings.
    * Per-instance JSON is None-safe: a missing heuristic / optimal baseline
      yields null rather than aborting the suite (mirrors the lost-sales suite).
    * Optimal DP is opt-in (--with_optimal_dp): it is slow on the l_r=3,4 rows and
      must never block a resumable launch; the best heuristic (capped_dual_index,
      ~0% published gap) is the optimal proxy when the DP is off.

ALGORITHM (per instance)
    1. Pull instance params + protocol from the Rust grid expansion.
    2. Compute the four heuristic costs on a fixed demand path (Rust search).
    3. (opt-in) Solve the bounded-DP optimum.
    4. Train each soft-tree spec with CMA-ES (run_experiment) and evaluate cost.
    5. Record learned cost, gap vs best heuristic, gap vs optimal (when present),
       and the published Figure-9 gaps. Aggregate across instances at the end.

USAGE
    python scripts/dual_sourcing/benchmark_full_suite.py \
        --run_tag dual_sourcing_paper_suite --budget full \
        --mp_num_processors 4 --instance_jobs 1 --reuse_existing
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dual_sourcing_benchmark_lib as lib

from invman.experiment_runner import run_experiment
from invman.utils import RunStatusTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the dual-sourcing literature-aligned benchmark suite over the Gijsbrechts Figure-9 instance family."
    )
    parser.add_argument("--grid_name", default=lib.GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME)
    parser.add_argument("--run_tag", default="dual_sourcing_paper_suite")
    parser.add_argument("--budget", choices=sorted(lib.COMMON_BUDGET), default=lib.DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--eval_seeds", type=int, default=None)
    parser.add_argument("--training_episodes", type=int, default=None)
    parser.add_argument("--training_horizon", type=int, default=None)
    parser.add_argument("--references", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Optional subset of experiment ids to run.")
    parser.add_argument("--with_optimal_dp", action="store_true", help="Also solve the bounded-DP optimum per instance (slow on l_r=3,4).")
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
    return lib.get_budget_config(parsed.budget)


def _effective_eval_horizon(parsed) -> int:
    return int(parsed.eval_horizon if parsed.eval_horizon is not None else _budget(parsed)["eval_horizon"])


def _effective_eval_seeds(parsed) -> int:
    return int(parsed.eval_seeds if parsed.eval_seeds is not None else _budget(parsed)["eval_seeds"])


def _effective_training_episodes(parsed) -> int:
    return int(parsed.training_episodes if parsed.training_episodes is not None else _budget(parsed)["training_episodes"])


def _effective_training_horizon(parsed) -> int:
    return int(parsed.training_horizon if parsed.training_horizon is not None else _budget(parsed)["horizon"])


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
    path = lib.result_path_for(args)
    if reuse_existing and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    return run_experiment(args)


def _instance_protocol(parsed) -> dict:
    budget = _budget(parsed)
    return {
        "budget": parsed.budget,
        "training_episodes": _effective_training_episodes(parsed),
        "es_population": budget["es_population"],
        "es_population_sampling": budget.get("es_population_sampling", "fixed"),
        "es_population_candidates": budget.get("es_population_candidates"),
        "es_population_probabilities": budget.get("es_population_probabilities"),
        "training_horizon": _effective_training_horizon(parsed),
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
        "Benchmark: Gijsbrechts et al. (2022), Section 6.2 / Figure 9. Heuristics are",
        "grid-searched in Rust on a fixed demand path; the published per-instance",
        "optimality-gap labels (single/dual/capped-dual index, tailored base-surge, a3c)",
        "are carried as literature metadata.",
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
        "| Policy | Mean rel. gap vs best heuristic (%) | Mean rel. gap vs optimal (%) | <= best heuristic (count) | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for policy_id, item in summary["aggregate"]["policies"].items():
        gap_h = item["mean_relative_gap_pct_vs_best_heuristic"]
        gap_o = item["mean_relative_gap_pct_vs_optimal"]
        lines.append(
            f"| `{policy_id}` | "
            f"`{'n/a' if gap_h is None else format(gap_h, '.4f')}` | "
            f"`{'n/a' if gap_o is None else format(gap_o, '.4f')}` | "
            f"`{item['better_or_equal_best_heuristic_count']}/{item['instances_with_best_heuristic']}` | "
            f"`{item['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Instance Files",
            "",
            "Each per-instance JSON contains:",
            "",
            "- benchmark parameters and literature metadata",
            "- Rust heuristic evaluations (single/dual/capped-dual index, tailored base-surge)",
            "- optional bounded-DP optimum",
            "- published Figure-9 optimality-gap labels (including a3c)",
            "- learned-policy evaluations and relative gaps vs the best heuristic and optimal",
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize_instances(instances: list[dict]) -> dict:
    policies = {}
    for spec in lib.EXPERIMENT_SPECS:
        policy_id = spec["id"]
        costs = []
        rel_gaps_h = []
        rel_gaps_o = []
        better_count = 0
        instances_with_best = 0
        for instance in instances:
            if policy_id not in instance["learned_policies"]:
                continue
            learned_cost = instance["learned_policies"][policy_id]["evaluation"]["learned_policy"]["mean_cost"]
            costs.append(learned_cost)
            best_heuristic = instance["comparative_summary"]["best_heuristic_cost"]
            if best_heuristic is not None:
                instances_with_best += 1
                rel_gaps_h.append(100.0 * (learned_cost - best_heuristic) / best_heuristic)
                if learned_cost <= best_heuristic:
                    better_count += 1
            optimal_cost = instance["comparative_summary"].get("optimal_cost")
            if optimal_cost is not None:
                rel_gaps_o.append(100.0 * (learned_cost - optimal_cost) / optimal_cost)
        if not costs:
            continue
        policies[policy_id] = {
            "label": spec["label"],
            "status": spec.get("status", "candidate"),
            "num_instances": len(costs),
            "mean_cost_across_instances": float(sum(costs) / len(costs)),
            "mean_relative_gap_pct_vs_best_heuristic": float(sum(rel_gaps_h) / len(rel_gaps_h)) if rel_gaps_h else None,
            "mean_relative_gap_pct_vs_optimal": float(sum(rel_gaps_o) / len(rel_gaps_o)) if rel_gaps_o else None,
            "instances_with_best_heuristic": int(instances_with_best),
            "better_or_equal_best_heuristic_count": int(better_count),
        }
    return {"policies": policies}


def _build_instance_payload(parsed, root: Path, instance: dict, selected_ids: set[str] | None, tracker=None):
    reference_name = instance["name"]
    instance_summary_path = _instance_summary_path(root, reference_name)
    if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
        return json.loads(instance_summary_path.read_text(encoding="utf-8"))

    if tracker is not None:
        tracker.update("benchmarking_heuristics", reference_name=reference_name)

    # Heuristic baselines (Rust search on a fixed demand path) + optional DP optimum.
    benchmark_args = lib.build_reference_args(reference_name)
    benchmark_args.seed = int(instance["search"]["search_seed"])
    search_horizon = int(instance["search"]["search_horizon"])
    heuristics = lib.evaluate_default_heuristics(benchmark_args, seed=int(instance["search"]["search_seed"]), horizon=search_horizon)
    best_heuristic_name, best_heuristic_cost = lib.best_heuristic(heuristics)
    optimal = lib.bounded_dp_optimal(benchmark_args) if parsed.with_optimal_dp else {
        "mean_cost": None, "available": False, "source": "skipped (use --with_optimal_dp)"
    }
    optimal_cost = optimal.get("mean_cost")

    learned_policies = {}
    for spec in lib.EXPERIMENT_SPECS:
        if selected_ids is not None and spec["id"] not in selected_ids:
            continue
        if tracker is not None:
            tracker.update("running_policy", reference_name=reference_name, policy_id=spec["id"])
        args = lib.configure_run_args(parsed, spec, root, reference_name)
        # Honor suite-level budget overrides on the resumable per-instance args.
        args.eval_horizon = _effective_eval_horizon(parsed)
        args.eval_seeds = _effective_eval_seeds(parsed)
        args.training_episodes = _effective_training_episodes(parsed)
        args.horizon = _effective_training_horizon(parsed)
        payload, result_path = _load_or_run_experiment(args, reuse_existing=parsed.reuse_existing)
        learned_policies[spec["id"]] = {
            "results_path": str(result_path),
            "policy_spec": spec,
            "evaluation": payload["evaluation"],
            "payload": payload,
            "checkpoint_glob": str((root / "models" / f"{args.experiment_name}_*").resolve()),
        }

    comparative = {
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "optimal_cost": optimal_cost,
        "policy_gaps": {},
    }
    for policy_id, item in learned_policies.items():
        learned_cost = float(item["evaluation"]["learned_policy"]["mean_cost"])
        comparative["policy_gaps"][policy_id] = {
            "gap_vs_best_heuristic": None if best_heuristic_cost is None else learned_cost - best_heuristic_cost,
            "relative_gap_pct_vs_best_heuristic": None if best_heuristic_cost is None else 100.0 * (learned_cost - best_heuristic_cost) / best_heuristic_cost,
            "gap_vs_optimal": None if optimal_cost is None else learned_cost - optimal_cost,
            "relative_gap_pct_vs_optimal": None if optimal_cost is None else 100.0 * (learned_cost - optimal_cost) / optimal_cost,
        }

    published = dict(instance["literature_metadata"].get("published_optimality_gap_pct", {}))
    published.pop("source", None)
    published.pop("url", None)

    payload = {
        "reference_instance": reference_name,
        "reference_description": instance["description"],
        "params": instance["params"],
        "search": instance["search"],
        "evaluation_protocol": instance["evaluation"],
        "literature_metadata": instance["literature_metadata"],
        "protocol": _instance_protocol(parsed),
        "heuristics": heuristics,
        "optimal": optimal,
        "published_optimality_gap_pct": published,
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
    if parsed.training_episodes is not None:
        command.extend(["--training_episodes", str(parsed.training_episodes)])
    if parsed.training_horizon is not None:
        command.extend(["--training_horizon", str(parsed.training_horizon)])
    if parsed.same_seed:
        command.append("--same_seed")
    if parsed.with_optimal_dp:
        command.append("--with_optimal_dp")
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
        grid = lib.get_benchmark_grid(parsed.grid_name)
        grid_instances = lib.build_grid_instances(parsed.grid_name)
        if parsed.references is not None:
            requested = set(parsed.references)
            grid_instances = [instance for instance in grid_instances if instance["name"] in requested]
        if parsed.limit is not None:
            grid_instances = grid_instances[: int(parsed.limit)]

        selected_ids = set(parsed.only) if parsed.only else None
        instance_payloads = _collect_instance_payloads(
            parsed, root, grid_instances, selected_ids, tracker=tracker
        )

        if parsed.skip_suite_summary:
            return

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
                for spec in lib.EXPERIMENT_SPECS
                if selected_ids is None or spec["id"] in selected_ids
            ],
            "instances": instance_payloads,
            "aggregate": _summarize_instances(instance_payloads),
        }

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
