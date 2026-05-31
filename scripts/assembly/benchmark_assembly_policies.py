#!/usr/bin/env python3
"""Benchmark for multi_echelon/assembly: optimal echelon base-stock vs heuristics.

OBJECTIVE
---------
The assembly problem is literature-verified (Rosling 1989 reduction to a serial system +
the Clark-Scarf serial solver that reproduces published optima). For this problem the
OPTIMAL policy is known analytically: the echelon base-stock policy at the Rosling
serial-equivalent levels. There is no learned-policy Python binding for `assembly` in the
installed invman_rust (the module is not registered in multi_echelon/bindings.rs and has
no rollout binding), so a learned soft-tree benchmark cannot be run without a Rust rebuild
(which is out of scope for this agent). See the BLOCKER note in the report.

What this DOES benchmark (all runnable now, no Rust):
  - OPTIMAL: echelon base-stock at the exact Rosling/Clark-Scarf levels (the analytic
    optimum; this IS the policy a learned method would have to beat / match).
  - HEURISTIC 1 (myopic newsvendor kit): set the finished echelon level to the
    single-stage newsvendor for the finished stage and the kit level to cover the kit
    lead-time demand at the same critical ratio (a natural decentralized base-stock guess).
  - HEURISTIC 2 (mean lead-time-demand): echelon levels = mean demand over the cumulative
    lead time at each stage (no safety stock) -- a deliberately weak baseline.

It reuses the faithful Python reimplementation in
verify_assembly_rosling_independent.py (already independently validated to reproduce the
published serial anchors and the assembly verification.rs instances).

The instance set is the three verification.rs instances (the problem's literature anchor
set) plus two additional in-scope instances (finished lead time 1) for breadth.
"""

import sys
import os
import math
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_assembly_rosling_independent import (  # noqa: E402
    AssemblyConfig, reduce_equal_lead_time, solve_from_local_costs,
    simulate_assembly, rel_err,
)


def poisson_newsvendor_oul(mean, periods, critical_ratio):
    """Smallest integer S with P(D_periods <= S) >= critical_ratio, D ~ Poisson(mean*periods)."""
    m = mean * max(periods, 0)
    if m <= 0:
        return 0.0
    s = 0
    while poisson.cdf(s, m) < critical_ratio and s < 10000:
        s += 1
    return float(s)


def heuristic_myopic_newsvendor_levels(config, demand):
    """Decentralized newsvendor: finished echelon = newsvendor over finished lead time at
    critical ratio p/(p+h_finished); kit echelon = finished level + newsvendor over the
    component lead time at the kit critical ratio p/(p+kit_holding). A natural but
    suboptimal base-stock heuristic (ignores the Clark-Scarf induced-penalty coupling)."""
    mean = demand[1]
    p = config.penalty
    cr_fin = p / (p + config.finished_holding_cost)
    cr_kit = p / (p + config.kit_holding_cost)
    # finished echelon covers finished lead-time demand (L_a) plus this period
    s_finished = poisson_newsvendor_oul(mean, config.finished_lead_time + 1, cr_fin)
    # kit echelon adds coverage of the component lead-time demand
    s_kit = s_finished + poisson_newsvendor_oul(mean, config.component_lead_time, cr_kit)
    return [s_finished, s_kit]


def heuristic_mean_cover_levels(config, demand):
    """Zero-safety-stock baseline: echelon levels = mean demand over cumulative lead time."""
    mean = demand[1]
    s_finished = mean * (config.finished_lead_time + 1)
    s_kit = s_finished + mean * config.component_lead_time
    return [s_finished, s_kit]


def benchmark_instance(name, config, demand, seed, periods=200_000, warm_up=5_000):
    assert config.finished_lead_time == 1, "in-scope: finished lead time 1"
    local_ud, lead_ud, penalty = reduce_equal_lead_time(config)
    opt_levels, exact_cost = solve_from_local_costs(local_ud, lead_ud, penalty, demand)

    policies = {
        "OPTIMAL (echelon base-stock, Clark-Scarf)": opt_levels,
        "HEUR myopic-newsvendor": heuristic_myopic_newsvendor_levels(config, demand),
        "HEUR mean-cover (no safety stock)": heuristic_mean_cover_levels(config, demand),
    }

    print(f"### {name}")
    print(f"    components={config.component_holding_costs} L_c={config.component_lead_time} "
          f"h_fin={config.finished_holding_cost} L_a={config.finished_lead_time} p={config.penalty} "
          f"demand={demand}")
    print(f"    exact optimal cost = {exact_cost:.4f}  (analytic Clark-Scarf optimum)")
    rows = []
    for label, levels in policies.items():
        avg, hold, back, _ = simulate_assembly(config, demand, levels, periods, warm_up, seed)
        gap = (avg - exact_cost) / exact_cost
        rows.append((label, levels, avg, hold, back, gap))
        levels_str = "[" + ", ".join("%.1f" % x for x in levels) + "]"
        print(f"    {label:42s} levels={levels_str:>16s} "
              f"cost={avg:8.4f}  gap_vs_opt={gap:+7.2%}")
    print()
    return exact_cost, rows


def main():
    instances = [
        # The three assembly/verification.rs literature-anchor instances:
        ("V1 2-comp Poisson(5) [anchor]", AssemblyConfig([1.0, 1.0], 1, 3.0, 1, 10.0), ("poisson", 5.0), 3),
        ("V2 3-comp L_c=2 Poisson(5) [anchor]", AssemblyConfig([1.0, 1.0, 1.0], 2, 7.0, 1, 37.12), ("poisson", 5.0), 7),
        ("V3 heterogeneous comp Poisson(4) [anchor]", AssemblyConfig([0.5, 1.5], 2, 4.0, 1, 20.0), ("poisson", 4.0), 11),
        # Two additional in-scope instances for breadth (finished lead time 1):
        ("E1 4-comp L_c=1 Poisson(8)", AssemblyConfig([0.5, 0.5, 0.5, 0.5], 1, 4.0, 1, 19.0), ("poisson", 8.0), 23),
        ("E2 2-comp L_c=3 Poisson(3)", AssemblyConfig([1.0, 1.0], 3, 5.0, 1, 25.0), ("poisson", 3.0), 31),
    ]
    print("=" * 78)
    print("ASSEMBLY POLICY BENCHMARK (independent Python env; in-scope: finished lead time 1)")
    print("=" * 78)
    print()
    summary = []
    for name, cfg, dem, seed in instances:
        exact_cost, rows = benchmark_instance(name, cfg, dem, seed)
        summary.append((name, exact_cost, rows))

    print("=" * 78)
    print("SUMMARY: gap of each policy vs the analytic Clark-Scarf optimum (lower = better)")
    print("=" * 78)
    labels = [r[0] for r in summary[0][2]]
    header = f"{'instance':38s}" + "".join(f"{l.split('(')[0].split()[0][:10]:>12s}" for l in labels)
    print(header)
    for name, exact_cost, rows in summary:
        line = f"{name:38s}"
        for (label, levels, avg, hold, back, gap) in rows:
            line += f"{gap:+11.2%} "
        print(line)


if __name__ == "__main__":
    main()
