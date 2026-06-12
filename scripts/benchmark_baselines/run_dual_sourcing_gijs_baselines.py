#!/usr/bin/env python
"""Executable baseline report for `dual_sourcing` (Gijsbrechts 2022 Figure-9).

OBJECTIVE
    Demonstrate the executable baseline layer on the 6 published Gijsbrechts et
    al. (2022) Section 6.2 / Figure-9 dual-sourcing instances
    (`dual_l{2,3,4}_ce{105,110}`). For each instance it prints the published
    optimality GAPS (capped-dual-index 0%, tailored-base-surge 0.06%,
    dual-index 0.11%, single-index 0.56%, A3C 0.52%) and — with `--simulate` —
    grid-searches the four heuristics on a fixed demand path to recover the
    ABSOLUTE costs (capped-dual-index is the ~0%-gap optimal proxy / number to
    beat).

ALGORITHM
    Thin wrapper over `benchmark_baseline_report.run("dual_sourcing")`: loads
    each instance through `invman.benchmarks.runners`, reads the params +
    published gaps, optionally calls `run_baselines` (the Rust
    `*_search_from_demands` bindings), and renders the comparison table.

USAGE
    python scripts/benchmark_baselines/run_dual_sourcing_gijs_baselines.py --simulate
    python scripts/benchmark_baselines/run_dual_sourcing_gijs_baselines.py --simulate --full --out outputs/benchmark_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from benchmark_baseline_report import run  # noqa: E402

if __name__ == "__main__":
    run("dual_sourcing")
