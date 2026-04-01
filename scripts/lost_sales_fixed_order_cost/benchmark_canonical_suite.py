import argparse
import json
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
from invman.problems.lost_sales_fixed_order_cost.reference_instances import get_reference_instance


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


def _render_markdown(summary):
    heuristic = summary["heuristics"]["evaluation"]
    best_heuristic_cost = min(
        heuristic[name]["mean_cost"] for name in ("s_s", "s_nq", "modified_s_s_q")
    )
    optimal_reference = summary.get("optimal_reference")
    has_optimal = bool(optimal_reference and optimal_reference.get("available"))
    lines = [
        "# Fixed-Cost Benchmark Suite",
        "",
        f"Reference instance: `{summary['reference']}`",
        "",
    ]
    if has_optimal:
        lines.extend(
            [
                "## Literature Anchor",
                "",
                f"- optimal: `{optimal_reference['mean_cost']:.5f}`",
                "",
            ]
        )
    lines.extend(
        [
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
            (
                "| Approximator | Architecture | Backend | Eval horizon | Mean cost | Gap vs best heuristic | Gap vs optimal |"
                if has_optimal
                else "| Approximator | Architecture | Backend | Eval horizon | Mean cost | Gap vs best heuristic |"
            ),
            (
                "| --- | --- | --- | ---: | ---: | ---: | ---: |"
                if has_optimal
                else "| --- | --- | --- | ---: | ---: | ---: |"
            ),
        ]
    )
    for result in summary["learned_policies"]:
        learned_cost = result["evaluation"]["learned_policy"]["mean_cost"]
        row = [
            result["label"],
            f"`{result['payload']['policy_architecture']}`",
            f"`{result['payload']['rollout_backend']}`",
            f"`{result['payload']['evaluation_horizon']}`",
            f"`{learned_cost:.5f}`",
            f"`{learned_cost - best_heuristic_cost:.5f}`",
        ]
        if has_optimal:
            row.append(f"`{learned_cost - float(optimal_reference['mean_cost']):.5f}`")
        lines.append(
            "| " + " | ".join(
                [row[0], *row[1:]]
            ) + " |"
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
    return result_path_for(args)


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
    optimal_reference = (
        get_reference_instance(parsed.reference)
        .get("benchmark_anchors", {})
        .get("published_optimal_reference", {"available": False})
    )

    if parsed.reuse_existing_summary and summary_json.exists():
        existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        heuristic_summary = existing_summary["heuristics"]
        optimal_reference = existing_summary.get("optimal_reference", optimal_reference)
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
        "search_horizon": parsed.search_horizon,
        "eval_horizon": parsed.eval_horizon,
        "eval_seeds": parsed.eval_seeds,
        "optimal_reference": optimal_reference,
        "heuristics": heuristic_summary,
        "learned_policies": learned_policy_results,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


if __name__ == "__main__":
    main()
