# Experiments

Benchmark home for `procurement_removal_inventory`.

## Runnable benchmark

`scripts/procurement_removal_inventory/benchmark_procurement_removal.py` is the current,
self-contained benchmark. It uses only the installed `invman_rust` extension and `invman.cmaes`
(it does **not** import the removed `invman.policies.soft_tree`). It:

- reports the reduced exact-DP optimum vs the two carried heuristics
- grid-searches the best constant interval-stock and returnability-buffer policies on both the
  primary instance and the removal-active instance, scored by Monte-Carlo mean discounted cost
- optionally (`--train`) trains a CMA-ES soft-tree over the Rust population rollout and benchmarks it
  on the same held-out seeds

Latest results: `outputs/procurement_removal_inventory/benchmark_2026-05-31.json` (summarized in the
problem-level `README.md`).

## Instances

- `PRIMARY_REFERENCE_INSTANCE` — removal lever essentially inactive (demand drains inventory)
- `REMOVAL_ACTIVE_REFERENCE_INSTANCE` — removal lever binds (overstocked start)

## Legacy / broken scripts

`train_soft_tree_reference.py` and `validate_against_exact_dp.py` predate the Python-cleanup
migration. `validate_against_exact_dp.py` runs again after the `common.py` import fix (it uses only
heuristic / exact-DP helpers). `train_soft_tree_reference.py` still constructs the removed
`SoftTreePolicy` and will fail; use `benchmark_procurement_removal.py --train` instead.

## Code anchors

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
- learned-policy rollout in `rollout.rs` (soft-tree, 7-feature policy-side map)
