# Experiments

This folder is reserved for paper benchmark definitions for `joint_pricing_inventory`.

The intended comparator stack is:

- CMA-ES-tuned learned policies
- price-and-order heuristics
- exact finite-horizon DP on the reduced verification slice

## Runnable benchmark

The feasible benchmark (no Rust rebuild, no retrain) lives at
`scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py`. It reports:

- exact-DP-anchored profit optimality gaps on the verifier instance
  (heuristics: `static_price_base_stock` 2.02%, `inventory_sensitive_base_stock` 16.83%)
- learned soft-tree vs heuristics on the primary instance (soft tree beats the best heuristic by
  +25.15% profit on 4096 held-out seeds)

See the package `README.md` "Benchmark Results" for the full tables and the honest verification status.
