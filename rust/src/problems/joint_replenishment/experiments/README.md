# Experiments

This folder is the paper-facing benchmark home for `joint_replenishment`.

Planned use:

- define reported multi-item instances
- compare CMA-ES-optimized learned policies against the carried heuristics
- include the reduced exact DP comparator when tractable

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
- runnable benchmark script: `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`

Benchmark currently runnable WITHOUT a Rust rebuild (numbers reproduced 2026-05):

- literature anchor (setting 5, infinite-horizon value iteration): env optimal action at `(5,0)` is
  `q=(0,6)`, matching Vanvuchelen et al. (2020) Figure 3.
- repo reduced finite-horizon DP comparator (setting-1 family, 4 periods, discounted): optimal
  `(6,6)` cost `266.39`; MOQ `(7,5)` cost `386.10`; DYN-OUT `(6,6)` cost `383.96`.
- heuristic Monte-Carlo sweep over all 16 Table-2 settings (200 periods, 256 reps, discounted): MOQ
  is at or below DYN-OUT on every setting.

Learned soft-tree vs heuristics (run 2026-05-31, no Rust rebuild):

- Driver: `scripts/joint_replenishment/benchmark_learned_vs_heuristics.py` (CMA-ES via
  `invman.es_mp.train` + `invman.policy.Policy`, backbone `soft_tree`). The stale
  `invman.policies.soft_tree` import in `scripts/joint_replenishment/common.py` was migrated to the
  current `invman.policy.Policy` API; the helper now builds the model and rollout kwargs against the
  installed binding (no rebuild required).
- Budget: depth 2, oblique split, linear leaves, temperature 0.25, popsize 24, 120 CMA-ES generations,
  sigma0 1.5, train_seed_batch 4 (CRN within a generation, seeds advance each generation).
- Held-out eval protocol: 2048 paired common-random-number seeds from base 1_000_000 (disjoint from the
  training seed block at base 123); the same eval-seed block scores the learned policy and both
  heuristics. Horizon 200 periods, discounted gamma 0.99, initial inventory zeros, per-item action box
  2*truck_capacity. Heuristics use newsvendor order-up-to targets S_i = ceil-to cr_i = b_i/(b_i+h_i).
- CPU cap: RAYON_NUM_THREADS / OPENBLAS / OMP / MKL / NUMEXPR all pinned to 2, mp_num_processors 1.
- Outcome (mean discounted cost over 2048 held-out seeds): learned soft-tree beats the best heuristic
  on 6 of 16 settings, loses on 10. MOQ `(Q,S|T)` is the best heuristic on all 16 settings (DYN-OUT is
  dominated everywhere, confirming the earlier sweep). The learned policy wins where holding/shortage
  costs are low (h=1, b=19: settings 5, 6, 13, 14 -> +4.2% to +13.0%, plus the marginal low-cost
  settings 1 and 9), i.e. where truckload-timing flexibility pays off; it loses on the high-cost
  settings (h=5, b=95: settings 3, 4, 11, 12, 16 -> -8.9% to -18.1%) where ordering to a tight base
  stock every period is near-optimal and MOQ matches it with less action variance. Full per-setting
  table below.

| Setting | Learned | DYN-OUT | MOQ | Best heur | Gap (best-learned) | %win | Winner |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| setting_1 | 5993.62 | 6152.10 | 6024.07 | MOQ | +30.45 | +0.51% | learned |
| setting_2 | 7645.45 | 7769.90 | 7186.22 | MOQ | -459.23 | -6.39% | MOQ |
| setting_3 | 8367.58 | 8164.73 | 7470.30 | MOQ | -897.27 | -12.01% | MOQ |
| setting_4 | 10197.08 | 8885.12 | 8632.45 | MOQ | -1564.63 | -18.13% | MOQ |
| setting_5 | 6605.41 | 7827.37 | 7596.67 | MOQ | +991.27 | +13.05% | learned |
| setting_6 | 8388.64 | 9423.17 | 8758.82 | MOQ | +370.18 | +4.23% | learned |
| setting_7 | 9180.72 | 9800.23 | 9042.90 | MOQ | -137.81 | -1.52% | MOQ |
| setting_8 | 10833.82 | 10560.39 | 10205.05 | MOQ | -628.77 | -6.16% | MOQ |
| setting_9 | 5940.71 | 6058.65 | 6005.06 | MOQ | +64.35 | +1.07% | learned |
| setting_10 | 7100.84 | 7413.05 | 7058.78 | MOQ | -42.06 | -0.60% | MOQ |
| setting_11 | 8699.72 | 8316.01 | 7625.03 | MOQ | -1074.69 | -14.09% | MOQ |
| setting_12 | 9838.53 | 8741.57 | 8678.75 | MOQ | -1159.78 | -13.36% | MOQ |
| setting_13 | 6774.09 | 7727.34 | 7648.80 | MOQ | +874.71 | +11.44% | learned |
| setting_14 | 8141.37 | 9080.87 | 8702.53 | MOQ | +561.16 | +6.45% | learned |
| setting_15 | 9543.87 | 9976.12 | 9268.77 | MOQ | -275.10 | -2.97% | MOQ |
| setting_16 | 11243.73 | 10410.25 | 10322.50 | MOQ | -921.23 | -8.92% | MOQ |

- Literature reference (anchor, not an absolute-cost benchmark): on setting 5, the published Figure-3
  optimal action at state (5,0) is q=(0,6) and both paper heuristics order q=(2,4)
  (`joint_replenishment_published_action_anchor`). The paper reports per-setting optimality only as a
  figure (Figure 2: heuristics 4-25% above optimal), so absolute optimal costs are not asserted
  per-setting. The learned policy's 13.0% improvement over MOQ on setting 5 is consistent with that
  4-25% heuristic optimality gap being partially recoverable by a learned policy.
- Raw JSON: `outputs/joint_replenishment/learned_vs_heuristics_vanvuchelen.json`.
