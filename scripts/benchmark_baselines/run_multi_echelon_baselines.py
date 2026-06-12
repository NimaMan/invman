#!/usr/bin/env python
"""Executable baseline report for `multi_echelon` (divergent: Van Roy / Gijs).

OBJECTIVE
    Demonstrate the executable baseline layer on the 5 divergent one-warehouse /
    N-retailer instances (`van_roy1997_simple_problem`, `..._case_study1/2`,
    `gijsbrechts2022_setting1/2`) — including the Gijs settings the user pointed
    at. For each instance it prints the published numbers (absolute constant
    base-stock + best-NDP cost on the Van Roy reproduction rows; A3C %-savings on
    the Gijs settings) and — with `--simulate` — grid-searches the best CONSTANT
    base-stock on the live env (the canonical comparator a learned policy beats),
    widening the grid to span the published optimum on the Van Roy rows.

ALGORITHM
    Thin wrapper over `benchmark_baseline_report.run("multi_echelon")`: loads
    each instance through `invman.benchmarks.runners`, reads params + published
    baselines, optionally calls `run_baselines`
    (`multi_echelon_search_stationary_policy`), and renders the table.

NOTE
    The `multi_echelon` umbrella also has serial / assembly / general-backorder-
    fixed-cost / PADN subfamilies with their own accessors; this report covers
    the divergent subfamily exposed by `multi_echelon_list_reference_instances`.
    The `--full` protocol is moderate (horizon 10000, ~30 replications); the
    fully faithful search uses each instance's `benchmark_search_horizon` /
    `benchmark_replications` (10000 x 100) — pass them through the runner for an
    exact reproduction.

USAGE
    python scripts/benchmark_baselines/run_multi_echelon_baselines.py --simulate
    python scripts/benchmark_baselines/run_multi_echelon_baselines.py --simulate --full --out outputs/benchmark_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from benchmark_baseline_report import run  # noqa: E402

if __name__ == "__main__":
    run("multi_echelon")
