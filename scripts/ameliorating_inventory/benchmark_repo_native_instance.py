#!/usr/bin/env python3
"""
Benchmark script for the ameliorating_inventory problem.

OBJECTIVE
---------
Compare the policies that are runnable from the *already installed* invman_rust
extension on the repo-native PRIMARY_REFERENCE_INSTANCE, without rebuilding Rust.

HONEST SCOPE / WHY THIS IS NOT LITERATURE-COMPARABLE
----------------------------------------------------
The cited paper (Pahr and Grunow 2025, "The Value of Blending", POM) uses a
materially richer model than the current Rust env:
  - objective: long-run AVERAGE PROFIT (paper) vs finite-horizon DISCOUNTED COST (Rust)
  - action: purchasing + production + issuance (paper) vs purchase-only (Rust)
  - stochastic purchase price, stochastic sales price (copula-correlated demand),
    stochastic age-dependent beta decay, evaporation, processing capacity (paper)
    vs fixed prices / fixed retention / no capacity (Rust)
  - 10 ages / 3 products generic, 25 ages port wine (paper) vs 5 ages / 2 products (Rust)
So the numbers below are INTERNALLY meaningful (apples-to-apples across the repo's
own policies on the repo's own instance) but are NOT comparable to any published row.

WHAT IS FEASIBLE NOW (installed bindings only)
----------------------------------------------
  - ameliorating_inventory_simulate_policy           (heuristic Monte-Carlo cost)
  - ameliorating_inventory_policy_rollout_from_paths (heuristic on fixed demand paths)
  - ameliorating_inventory_soft_tree_rollout / _population_rollout / _from_paths
    (learned soft-tree policy; needs trained flat_params -> see BLOCKER 2)

BLOCKERS (cannot run from Python without changes outside this problem dir)
-------------------------------------------------------------------------
  BLOCKER 1 (exact optimum): finite_horizon_dp::solve_optimal_policy and
    evaluate_named_heuristic are #[cfg(test)] only and have NO pyfunction binding.
    To benchmark heuristics/learned vs exact-optimal from Python you would add (in
    ameliorating_inventory/bindings.rs, this problem's own file, then rebuild Rust):
        #[pyfunction] fn ameliorating_inventory_exact_optimal_cost(...) -> (f64, usize)
        #[pyfunction] fn ameliorating_inventory_exact_heuristic_cost(name, ...) -> (f64, usize)
    wrapping finite_horizon_dp::{solve_optimal_policy, evaluate_named_heuristic}
    over an ExactVerificationReference built from the passed instance, and register
    them in register_py. (Rebuild is intentionally NOT done here.)
  BLOCKER 2 (learned soft-tree): there is no checked-in trained parameter vector for
    this problem, and training requires the CMA-ES pipeline. Heuristics-only is run here.

USAGE
-----
    python scripts/ameliorating_inventory/benchmark_repo_native_instance.py
"""

import sys

import invman_rust as ir

# PRIMARY_REFERENCE_INSTANCE (literature/references.rs) -- repo-native, NOT published.
INSTANCE = dict(
    inventory_by_age=[0, 0, 0, 0, 0],   # num_ages = 5
    demand_kinds=["poisson", "poisson"],
    demand_means=[10.0, 6.0],
    target_ages=[1, 3],
    product_prices=[300.0, 500.0],
    age_retention=[0.98, 0.98, 0.98, 0.98, 0.98],
    purchase_cost_per_unit=250.0,
    holding_cost_per_unit=25.0,
    decay_salvage_values=[50.0, 60.0, 70.0, 80.0, 90.0],
)

# Benchmark targets carried in references.rs for the primary instance.
NEWSVENDOR_TOTAL_TARGET = 24
TWO_D_TOTAL_TARGET = 24
TWO_D_YOUNG_TARGET = 8
TWO_D_YOUNG_CUTOFF = 1

PERIODS = 40
REPLICATIONS = 2000
SEED = 20260531
DISCOUNT = 0.99


def run_heuristics():
    common = dict(
        inventory_by_age=INSTANCE["inventory_by_age"],
        periods=PERIODS,
        demand_kinds=INSTANCE["demand_kinds"],
        demand_means=INSTANCE["demand_means"],
        target_ages=INSTANCE["target_ages"],
        product_prices=INSTANCE["product_prices"],
        age_retention=INSTANCE["age_retention"],
        purchase_cost_per_unit=INSTANCE["purchase_cost_per_unit"],
        holding_cost_per_unit=INSTANCE["holding_cost_per_unit"],
        decay_salvage_values=INSTANCE["decay_salvage_values"],
        replications=REPLICATIONS,
        seed=SEED,
        discount_factor=DISCOUNT,
    )
    rows = []
    nv_mean, nv_std = ir.ameliorating_inventory_simulate_policy(
        "newsvendor_purchase", [float(NEWSVENDOR_TOTAL_TARGET)], **common
    )
    rows.append(("newsvendor_purchase", nv_mean, nv_std))
    td_mean, td_std = ir.ameliorating_inventory_simulate_policy(
        "two_dimensional_order_up_to",
        [float(TWO_D_TOTAL_TARGET), float(TWO_D_YOUNG_TARGET), float(TWO_D_YOUNG_CUTOFF)],
        **common,
    )
    rows.append(("two_dimensional_order_up_to", td_mean, td_std))
    return rows


def main():
    print("ameliorating_inventory repo-native benchmark (NOT literature-comparable)")
    print(f"  instance: 5 ages / 2 products, Poisson demand {INSTANCE['demand_means']}")
    print(f"  horizon={PERIODS} reps={REPLICATIONS} seed={SEED} discount={DISCOUNT}")
    print()
    print(f"  {'policy':32s} {'disc.cost mean':>15s} {'cost std':>12s} {'disc.profit':>13s}")
    for name, mean, std in run_heuristics():
        print(f"  {name:32s} {mean:15.2f} {std:12.2f} {(-mean):13.2f}")
    print()
    print("BLOCKER 1: exact-optimal and exact-heuristic DP have no Python binding")
    print("           (finite_horizon_dp.rs is #[cfg(test)] only). Add pyfunctions in")
    print("           ameliorating_inventory/bindings.rs + rebuild to enable optimal-gap rows.")
    print("BLOCKER 2: no checked-in trained soft-tree params; learned-policy row needs CMA-ES.")
    print()
    print("NOTE: these numbers measure repo policies on a repo instance; they do not")
    print("      reproduce any Pahr-Grunow (2025) published row (see literature/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
