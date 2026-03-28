import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "lost_sales_fixed_order_cost" / "benchmark_full_suite.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "lost_sales_fixed_order_cost" / "export_full_grid_paper_table.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed-cost full-grid benchmark and refresh the paper TeX table."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run_tag", default="fixed_cost_full_grid_suite_5k_paperlike")
    parser.add_argument("--grid_name", default="literature_subset_poisson_mu5")
    parser.add_argument("--mp_num_processors", type=int, default=4)
    parser.add_argument("--instance_jobs", type=int, default=1)
    parser.add_argument("--search_horizon", type=int, default=10000)
    parser.add_argument("--eval_horizon", type=int, default=int(1e6))
    parser.add_argument("--eval_seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--same_seed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--references", nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--no_reuse_existing", action="store_true")
    parser.add_argument("--no_reuse_existing_instance_summary", action="store_true")
    return parser.parse_args()


def _run(command: list[str]):
    rendered = " ".join(shlex.quote(part) for part in command)
    print(rendered, flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main():
    args = parse_args()
    benchmark_command = [
        args.python_bin,
        str(BENCHMARK_SCRIPT),
        "--run_tag",
        args.run_tag,
        "--grid_name",
        args.grid_name,
        "--mp_num_processors",
        str(args.mp_num_processors),
        "--instance_jobs",
        str(args.instance_jobs),
        "--search_horizon",
        str(args.search_horizon),
        "--eval_horizon",
        str(args.eval_horizon),
        "--eval_seeds",
        str(args.eval_seeds),
        "--seed",
        str(args.seed),
    ]
    if args.same_seed:
        benchmark_command.append("--same_seed")
    if args.limit is not None:
        benchmark_command.extend(["--limit", str(args.limit)])
    if args.references:
        benchmark_command.extend(["--references", *args.references])
    if args.only:
        benchmark_command.extend(["--only", *args.only])
    if not args.no_reuse_existing:
        benchmark_command.append("--reuse_existing")
    if not args.no_reuse_existing_instance_summary:
        benchmark_command.append("--reuse_existing_instance_summary")

    suite_json = PROJECT_ROOT / "outputs" / "benchmarks" / args.run_tag / "fixed_cost_full_suite.json"
    export_command = [
        args.python_bin,
        str(EXPORT_SCRIPT),
        "--suite_json",
        str(suite_json),
        "--output",
        str(PROJECT_ROOT / "paper" / "generated" / "fixed_cost_full_grid_table.tex"),
    ]

    _run(benchmark_command)
    _run(export_command)


if __name__ == "__main__":
    main()
