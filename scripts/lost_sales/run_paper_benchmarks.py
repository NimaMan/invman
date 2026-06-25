import argparse
import json
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import cpu_limited_environ, normalize_args_cpu_limits

SELECTED_POLICIES = (
    "linear_soft_gated_direct_quantity",
    "nn_soft_gated_direct_quantity_h8_selu",
    "linear_soft_gated_ordinal_quantity",
    "nn_soft_gated_ordinal_quantity_h8_selu",
    "soft_tree_depth1_linear_leaf",
    "soft_tree_depth2_linear_leaf",
)

SUITES = {
    "vanilla": {
        "problem": "lost_sales",
        "script": "scripts/lost_sales/benchmark_full_suite.py",
        "grid_name": "xin2020_extended_lost_sales",
        "run_tag": "lost_sales_paper_suite_2k_scale20_seed42",
        "summary_json": "lost_sales_full_suite.json",
        "summary_md": "lost_sales_full_suite.md",
    },
    "fixed_cost": {
        "problem": "lost_sales_fixed_order_cost",
        "script": "scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py",
        "grid_name": "lost_sales_style_full_grid_mu5",
        "run_tag": "fixed_cost_paper_suite_2k_scale20_seed42",
        "summary_json": "fixed_cost_full_suite.json",
        "summary_md": "fixed_cost_full_suite.md",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the vanilla and fixed-cost lost-sales paper benchmark suites."
    )
    parser.add_argument(
        "--suite",
        choices=("all", "vanilla", "fixed_cost"),
        default="all",
        help="Which suite to run. The default runs vanilla first, then fixed-cost.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--instance_jobs", type=int, default=2)
    parser.add_argument("--training_episodes", type=int, default=2000)
    parser.add_argument("--training_horizon", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument("--state_scale", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--reuse_existing_instance_summary", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _suite_order(selected: str) -> list[str]:
    if selected == "all":
        return ["vanilla", "fixed_cost"]
    return [selected]


def _command(parsed, suite_name: str) -> list[str]:
    suite = SUITES[suite_name]
    command = [
        sys.executable,
        str(PACKAGE_ROOT / suite["script"]),
        "--grid_name",
        suite["grid_name"],
        "--run_tag",
        suite["run_tag"],
        "--seed",
        str(parsed.seed),
        "--mp_num_processors",
        str(parsed.mp_num_processors),
        "--instance_jobs",
        str(parsed.instance_jobs),
        "--training_episodes",
        str(parsed.training_episodes),
        "--eval_horizon",
        str(parsed.eval_horizon),
        "--eval_seeds",
        str(parsed.eval_seeds),
        "--state_scale",
        str(parsed.state_scale),
        "--only",
        *SELECTED_POLICIES,
    ]
    if parsed.training_horizon is not None:
        command.extend(["--training_horizon", str(parsed.training_horizon)])
    if parsed.limit is not None:
        command.extend(["--limit", str(parsed.limit)])
    if parsed.reuse_existing:
        command.append("--reuse_existing")
    if parsed.reuse_existing_instance_summary:
        command.append("--reuse_existing_instance_summary")
    return command


def _suite_outputs(suite_name: str) -> dict:
    suite = SUITES[suite_name]
    root = PACKAGE_ROOT / "outputs" / "benchmarks" / suite["run_tag"]
    return {
        "problem": suite["problem"],
        "grid_name": suite["grid_name"],
        "run_tag": suite["run_tag"],
        "summary_json": str(root / suite["summary_json"]),
        "summary_md": str(root / suite["summary_md"]),
        "instances_dir": str(root / "instances"),
        "results_dir": str(root / "results"),
        "models_dir": str(root / "models"),
    }


def main():
    parsed = parse_args()
    mp_num_processors = normalize_args_cpu_limits(parsed)
    plan = []
    for suite_name in _suite_order(parsed.suite):
        plan.append(
            {
                "suite": suite_name,
                "command": _command(parsed, suite_name),
                "outputs": _suite_outputs(suite_name),
            }
        )

    print(json.dumps({"paper_benchmark_plan": plan}, indent=2))
    if parsed.dry_run:
        return

    for item in plan:
        subprocess.run(
            item["command"],
            cwd=PACKAGE_ROOT,
            check=True,
            env=cpu_limited_environ(mp_num_processors),
        )


if __name__ == "__main__":
    main()
