#!/usr/bin/env python
"""Executable baseline report for ANY catalog family (generic dispatcher).

OBJECTIVE
    The three `run_<family>_baselines.py` scripts are convenience wrappers for the
    headline families; this one takes the problem name as its first argument so a
    user can pull the baseline report for ALL 14 catalog families through one
    entry point (`runners.available_runners()` is the live list).

ALGORITHM
    Pop the leading `<problem>` positional off argv, then hand the rest to the
    shared `benchmark_baseline_report.run(problem)` harness (which loads each
    reference instance via `invman.benchmarks.runners`, reads published baselines,
    optionally re-runs them on the live env with `--simulate`, and renders the
    comparison table).

USAGE
    python scripts/benchmark_baselines/run_family_baselines.py perishable_inventory --simulate
    python scripts/benchmark_baselines/run_family_baselines.py joint_replenishment
    python scripts/benchmark_baselines/run_family_baselines.py spare_parts_inventory --simulate --out outputs/benchmark_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from benchmark_baseline_report import run  # noqa: E402


def _main() -> None:
    from invman.benchmarks import runners

    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        avail = ", ".join(runners.available_runners())
        print(f"usage: run_family_baselines.py <problem> [--simulate --full --out ...]\n"
              f"problems: {avail}")
        raise SystemExit(2)
    problem = sys.argv.pop(1)  # consume before the shared harness's own argparse
    run(problem)


if __name__ == "__main__":
    _main()
