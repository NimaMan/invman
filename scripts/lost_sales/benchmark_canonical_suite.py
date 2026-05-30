import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.experiment_runner import run_experiment
from invman.lost_sales_benchmark import (
    COMMON_BUDGET,
    EXPERIMENT_SPECS,
    benchmark_reference_instance,
    configure_run_args,
    result_path_for,
)


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
