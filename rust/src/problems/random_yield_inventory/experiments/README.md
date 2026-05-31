# Experiments

Benchmark home for `random_yield_inventory`.

## Runnable benchmark (2026-05)

`scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py` compares all currently
available policies against each other and writes a JSON + markdown report. It is self-contained against
the installed `invman_rust` and the current `invman.policy.Policy` / `invman.es_mp.train` interface.

Two slices:

1. **Exact-DP slice** (implementation-verified optimum, capped action space, `VERIFICATION_PROBLEM_INSTANCE`):
   exact-optimal DP vs LIR vs WNH, with optimality gaps. Results:
   optimal `40.0599`; LIR `47.7138` (gap `7.65`, +19.1%); WNH `60.3936` (gap `20.33`, +50.8%).
2. **Simulation slice** (`PRIMARY_REFERENCE_INSTANCE`, uncapped env, 2000 held-out seeds): LIR vs WNH
   and, with `--train_soft_tree`, a CMA-ES-trained soft-tree. Results (depth 3, 600 ep, pop 32):
   soft-tree `196.66` (best), LIR `203.62` (+3.4%), WNH `222.44` (+11.6%).

## Status of the older scripts

`scripts/random_yield_inventory/{common.py, train_soft_tree_reference.py}` import
`invman.policies.soft_tree.SoftTreePolicy`, a module path that **no longer exists** in the package
(the soft-tree descriptor now lives in `invman/policy.py` as `Policy(backbone="soft_tree", ...)`).
Those scripts therefore cannot run as-is; the new
`benchmark_policies_vs_exact_and_heuristics.py` supersedes them and uses the current interface.

## Code anchors

- heuristics in `heuristics/` (LIR `linear_inflation.rs`, WNH `weighted_newsvendor.rs`)
- exact reduced benchmark in `finite_horizon_dp.rs`
- soft-tree rollout binding in `rollout.rs` / `bindings.rs`

## Remaining steps

- Recover the exact published WNH formula (Chen 2018 / Yan 2026) to settle whether the order should be
  inflated by `1/p`; only then update `heuristics/weighted_newsvendor.rs`.
- If a public Yan/Chen per-instance number is ever recovered, add it to `references.rs` and assert it
  to upgrade the status to literature-verified.
