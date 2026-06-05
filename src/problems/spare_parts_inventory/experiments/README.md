# Experiments

This folder is the paper-facing benchmark home for `spare_parts_inventory`.

Planned use:

- define reported spare-parts instances
- compare CMA-ES-optimized learned policies against carried heuristics
- include reduced exact comparators where tractable
- keep exact literature-validation benchmarks separate from the learned-policy repairable-control
  experiments when the formulation is not the same MDP family

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`

Benchmark entry point (no Rust rebuild required, uses the installed `invman_rust`):

- `scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py` emits all three
  benchmark blocks (Kranenburg Table 5.2 exact reproduction, repo-native exact DP
  self-consistency, learned soft-tree vs heuristics on the primary instance).

Learned-policy result (2026-05-31, 17-period primary repairable instance, discount 0.99,
held-out 4096-seed block 900000..904096; soft-tree weights loaded from the saved CMA-ES
artifact `outputs/spare_parts_inventory/retry_d2_t010_e300_s123.json`):

| Policy | Params | Mean discounted cost | Soft-tree improvement |
| --- | --- | ---: | ---: |
| `soft_tree` (depth 2, oblique, linear, T=0.10) | trained | 53.06 | — |
| best constant `base_stock` | S=6 | 53.78 | 1.34% |
| benchmark `base_stock` | S=5 | 62.99 | 15.77% |
| `lead_time_mean_cover` | buffer=1.0 | 92.95 | 42.92% |

The learned soft-tree beats the best STATIC base-stock level it could have memorized
(S=6) by 1.34% out of sample, i.e. it earns a genuine state-dependent edge rather than
just rediscovering a constant order-up-to level. To retrain (rather than re-evaluate the
saved artifact) use `train_soft_tree_reference.py`.
