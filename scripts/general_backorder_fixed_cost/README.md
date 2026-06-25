# Learned-policy result: Geevers set-1 general network (`geevers2023_general_set1`)

Committed results artifact for the first learned-policy row on the
`multi_echelon/general_backorder_fixed_cost` set-1 instance (Geevers, van Hezewijk & Mes 2024,
CardBoard Company general network: 4 warehouses + 5 retailers, Poisson(15), unit lead times,
backorders). Produced by
`scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py` against the
installed `invman_rust` (no rebuild). Live ledger / per-run JSON (incl. the trained 81-dim
parameter vector) live under the gitignored
`outputs/autoresearch/general_backorder_fixed_cost_autoresearch/`; this file is the committed
snapshot.

## Baselines (from `references.rs`, set 1)

- published constant node-base-stock benchmark = **10,467** at levels `[82,100,64,83,35,35,35,35,35]`.
- repo simulator reproduction of that benchmark = **~10,354.8** (gap -1.1%, 3 seeds x 500 reps) -- the keep/discard GATE.
- published PPO best = **8,714** (DRL target, reported alongside).

## Action geometry (the policy)

Soft-tree `vector_quantity` output (9 dims) read as the per-node order-up-to (base-stock)
TARGET levels via the binding's `node_base_stock_targets` action mode. State-independent =>
constant base-stock; state-dependent => richer class. Warm-started so generation 0 reproduces
the published benchmark (constant-leaf logit encoding of the published levels; split weights
= 0). Action box: warehouses [0,220], retailers [0,140].

## Result (full budget, 2,000 held-out CRN seeds, depth-2 oblique constant leaf, sigma 0.20)

| run | warm-start gen0 (= benchmark) | learned held-out | gap vs repo heuristic (10,354.8) | vs published 10,467 | vs PPO best 8,714 | verdict |
|---|---|---|---|---|---|---|
| CMA seed 123 | 10,378.6 +/- 10.6 | **8,034.8 +/- 17.6** | -2,320.0 (-22.4%) | -2,432.2 | -679.2 | beats |
| CMA seed 777 | 10,378.6 +/- 10.6 | **7,590.7 +/- 19.2** | -2,764.1 (-26.7%) | -2,876.3 | -1,123.3 | beats |

Both independent CMA seeds beat the constant node-base-stock benchmark by >22% (>> 2x SEM,
genuine out-of-sample) and land below the published PPO best -- robust to initialization.
Generation 0 reproduces the benchmark, confirming the warm-start, so the ~2,300-2,800 cost
improvement is what CMA-ES added.

Headline: **learned policy beats the published constant base-stock benchmark by ~22-27% and
surpasses the published PPO best (8,714) by 679-1,123** on Geevers set 1.

## Seed-robust runner (the headline-producing entry point)

`scripts/general_backorder_fixed_cost/seed_robust_general_backorder.py` is the seed-ROBUST
runner backing the paper's 5-seed claim (learned 7,837.0 +/- 189.7, -24.3% +/- 1.8% vs the
reproduced benchmark, all 5 seeds below). It reuses the autoresearch entry point's helpers
verbatim (importlib; no env/policy/Rust changes), loops >= 5 independent CMA-ES optimizer
seeds (canonical 9001..9005 by default, `--seeds` to override), and delegates aggregation +
verdict to `invman/optimizer_seed_robustness_policy.py` (`srp.run_over_seeds`, sample n-1 std,
shared ROBUST_BEAT/PARITY/LOSS rule).

- GATE = the autoresearch keep/discard gate: repo reproduction of the published constant
  node-base-stock benchmark (`simulate_base_stock`, 3 sim seeds x 500 reps, ~10,354.8);
  optimizer-seed independent, so `gate_seed_std == 0`. The published PPO best (8,714) is
  carried as cross-protocol CONTEXT only.
- Real artifact: `outputs/general_backorder_fixed_cost/seed_robust_report.json` (standardized
  srp summary keys + per_seed records).
- `--smoke`: tiny plumbing test (existing "smoke" budget preset: popsize 8, 8 generations,
  4 train seeds, 64 eval seeds; gate at 3x20 reps; 1 worker) writing ONLY to
  `outputs/general_backorder_fixed_cost/smoke_seed_robust/seed_robust_report_smoke.json`.
- Full run (CPU-capped):
  `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python
  scripts/general_backorder_fixed_cost/seed_robust_general_backorder.py --budget full
  --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2`
