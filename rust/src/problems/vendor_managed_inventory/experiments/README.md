# Experiments

This folder is reserved for paper benchmark definitions for `vendor_managed_inventory`.

The intended comparator stack is:

- CMA-ES-tuned learned policies (soft decision tree)
- vendor-managed shipment heuristics (`retailer_base_stock`, `dc_reserve_base_stock`)
- exact finite-horizon DP on the reduced verification slice

## Current benchmark (reduced single-retailer slice)

Because the headline Giannoccaro & Pontrandolfo (2010) 8-case profit table is not reproducible from
the public text (see `../literature/README.md`), the runnable policy benchmark lives on the
repo-native reduced single-retailer slice (`env::step_state`), which is the env exposed to Python and
validated by the exact DP regression.

Runner (pure Python, no Rust rebuild; drives the installed `invman_rust` bindings + pycma):

- `scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py`

What it compares, on a held-out common-random-number seed bank, over `PRIMARY_REFERENCE_INSTANCE`
plus four perturbations (low/high stockout penalty, low/high demand):

- tuned `retailer_base_stock` (grid over base-stock level)
- tuned `dc_reserve_base_stock` (grid over level x reserve)
- CMA-ES soft decision tree (depth 2, 28 params, scalar shipment action)

### Result (held-out discounted cost, lower is better)

On this slice the optimal policy is essentially a base-stock threshold: the heuristic cost is convex
in the base-stock level with a clean single optimum, so the tuned base-stock heuristic is near
optimal. The CMA-ES soft tree learns an approximately base-stock-like policy and lands within ~1-3%
of the tuned heuristic but does not beat it. This is the honest, expected outcome for a single-stage
lost-sales slice with no extra structure to exploit.

The headline numbers and the exact per-instance table are printed by the script; re-run to refresh.

### Missing ceiling (blocker)

The exact finite-horizon DP optimal (`finite_horizon_dp::solve_optimal_policy`) is the correct ceiling
for this benchmark, and `verification/tests.rs` already proves it dominates both heuristics on the
small verifier instance. It is NOT exposed as a Python binding, so it cannot be added as a benchmark
column without a Rust rebuild + a `bindings.rs` edit. Exposing
`vendor_managed_inventory_solve_optimal_policy` is the top next step.
