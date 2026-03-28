import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.problems.lost_sales.benchmark import benchmark_reference_instance
from invman.problems.lost_sales.experiment_spec import (
    COMMON_BUDGET,
    EXPERIMENT_SPECS,
    configure_run_args,
    result_path_for,
)
from invman.problems.lost_sales.reference_instances import get_benchmark_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full vanilla lost-sales literature-aligned benchmark suite over the configured instance grid."
    )
    parser.add_argument("--grid_name", default="xin2020_extended_lost_sales")
    parser.add_argument("--run_tag", default="lost_sales_full_grid_suite_paperlike")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument("--references", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--reuse_existing_instance_summary", action="store_true")
    return parser.parse_args()


def _suite_root(run_tag: str) -> Path:
    return PACKAGE_ROOT / "outputs" / "benchmarks" / run_tag


def _ensure_dirs(root: Path):
    for dirname in ("results", "logs", "models", "instances"):
        (root / dirname).mkdir(parents=True, exist_ok=True)


def _summary_paths(root: Path):
    return root / "lost_sales_full_suite.json", root / "lost_sales_full_suite.md"


def _instance_summary_path(root: Path, reference_name: str) -> Path:
    return root / "instances" / f"{reference_name}.json"


def _load_or_run_experiment(args, *, reuse_existing: bool):
    path = result_path_for(args)
    if reuse_existing and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    return run_experiment(args)


def _instance_protocol(parsed) -> dict:
    return {
        "training_episodes_default": COMMON_BUDGET["training_episodes_default"],
        "training_episodes_lead_time_2": COMMON_BUDGET["training_episodes_lead_time_2"],
        "es_population": COMMON_BUDGET["es_population"],
        "training_horizon_default": COMMON_BUDGET["horizon_default"],
        "training_horizon_lead_time_2": COMMON_BUDGET["horizon_lead_time_2"],
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
        f"- training episodes (default): `{summary['protocol']['training_episodes_default']}`",
        f"- training episodes for `L=2`: `{summary['protocol']['training_episodes_lead_time_2']}`",
        f"- ES population: `{summary['protocol']['es_population']}`",
        f"- training horizon (default): `{summary['protocol']['training_horizon_default']}`",
        f"- training horizon for `L=2`: `{summary['protocol']['training_horizon_lead_time_2']}`",
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
            "mean_relative_gap_pct_vs_best_heuristic": float(sum(rel_gaps) / len(rel_gaps)),
            "better_than_best_heuristic_count": int(better_count),
            "better_than_literature_cbs_count": int(better_than_cbs),
            "instances_with_literature_cbs": int(instances_with_cbs),
        }
    return {"policies": policies}


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
    instance_payloads = []
    for instance in grid_instances:
        reference_name = instance["name"]
        instance_summary_path = _instance_summary_path(root, reference_name)
        if parsed.reuse_existing_instance_summary and instance_summary_path.exists():
            instance_payloads.append(json.loads(instance_summary_path.read_text(encoding="utf-8")))
            continue

        heuristic_summary = benchmark_reference_instance(
            reference_name,
            eval_horizon=parsed.eval_horizon,
            eval_seeds=parsed.eval_seeds,
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
            "heuristics": heuristic_summary,
            "literature_references": {
                "optimal": heuristic_summary["optimal_reference"],
                "capped_base_stock": heuristic_summary["capped_base_stock_reference"],
            },
            "learned_policies": learned_policies,
            "comparative_summary": comparative,
        }
        instance_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        instance_payloads.append(payload)

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

    summary_json, summary_md = _summary_paths(root)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


if __name__ == "__main__":
    main()
