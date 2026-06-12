#!/usr/bin/env python
"""Executable baseline report for `lost_sales` (vanilla + fixed_order_cost).

OBJECTIVE
    Demonstrate the executable baseline layer on the family the user has "quite
    some problems" in: the 33-cell Zipkin vanilla grid plus the Bijvank (2015)
    fixed-order-cost instance. For every reference instance it prints the
    published baselines (optimal / myopic1 / myopic2 / svbs / capped for vanilla;
    optimal-DP / (s,S) / (s,nQ) / modified for fixed-cost) and — with
    `--simulate` — re-runs them on the live env so the published numbers are
    reproduced in adjacent columns.

ALGORITHM
    Thin wrapper over `benchmark_baseline_report.run("lost_sales")`: that loads
    each instance through `invman.benchmarks.runners`, reads the params +
    published baselines, optionally calls `run_baselines` (vanilla ->
    `lost_sales_heuristics_all`; fixed -> the exact average-cost VI summary), and
    renders a markdown + JSON comparison table.

USAGE
    python scripts/benchmark_baselines/run_lost_sales_baselines.py
    python scripts/benchmark_baselines/run_lost_sales_baselines.py --simulate --full --out outputs/benchmark_baselines
    python scripts/benchmark_baselines/run_lost_sales_baselines.py --instances lit_poisson_p4_l4 bijvank2015_table1_l2_p14_k5 --simulate
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling report library and the `invman` package importable when the
# script is run directly from anywhere.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from benchmark_baseline_report import run  # noqa: E402

if __name__ == "__main__":
    run("lost_sales")
