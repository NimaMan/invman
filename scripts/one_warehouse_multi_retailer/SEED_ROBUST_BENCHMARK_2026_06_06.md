# OWMR seed-robust benchmark entries (2026-06-06)

> Finalized, seed-robust replacement for the two at-risk single-seed OWMR rows in the paper
> (`tab:owmr-results`). Format = **baseline (gate) | policy (design + seed-mean ± std over ≥5
> independent CMA seeds + artifact) | honest verdict vs the same-protocol gate | PPO as
> cross-protocol context**. No single-seed or best-of-N headline.
> Protocol: held-out 4096-path paired CRN, 100-period total cost, honest deployment floor
> (deploy ≥ gate). Same env/CRN blocks as `run_asymmetric_learned_vs_gate.py`.

## Why this supersedes the paper's current OWMR rows
The paper reports het-3 as **+1.33%** (one depth-2 seed) and high-CV-10 as **+6.44%** (one seed).
The campaign + A0 ablation showed the same het-3 instance spans 1108–1167 across seeds (the
single-seed number is a draw, not the expected value), and the audit flagged the high-CV-10 win as
a single-seed paper-table claim. Both are re-reported below as **6-seed means**, and the headline
shifts from the seed-fragile PPO-scalar comparison to the **robust same-protocol gate beat**.

## Row 1 — heterogeneous K=3 (`kaynov2024_instance_12`)

- **Baseline (reproducible):** tuned echelon base-stock + allocation **gate** `W=39, R=[5,10,1]` →
  **1169.59 ± 2.05** (deterministic grid search; held-out 4096-path CRN).
- **Policy (replicable, seed-robust):** depth-2 soft tree, `echelon_targets_with_alloc_targets`
  action head + `absolute_augmented` state, axis-aligned splits, linear leaves, **warm-started at
  the gate**, σ=0.05, pop 32 × 800 gen, train_seed_batch 24 (CRN). 6 optimizer seeds
  {841,842,844,845,846,847}:

  | seed | 841 | 842 | 844 | 845 | 846 | 847 | **mean ± std** |
  |---|---|---|---|---|---|---|---|
  | cost | 1125.85 | 1109.21 | 1114.84 | 1114.14 | 1114.30 | 1114.31 | **1115.44 ± 5.51** |
  | vs gate | +3.74% | +5.16% | +4.68% | +4.74% | +4.73% | +4.73% | **+4.63%** |

- **Verdict:** **robustly beats the same-protocol gate by +4.63%** (all 6 seeds beat it; mean−std =
  +4.16% still clears the gate). Vs the published Kaynov **PPO scalar 1118.92** (cross-protocol):
  seed-mean +0.31% below, 5/6 seeds below — i.e. it **straddles** PPO within seed noise, so **no
  robust PPO beat is claimed** (PPO is context only).
- **Artifact:** per-seed `outputs/owmr_ppo_campaign/ablation_depth/i12_d2_warmgate_s0.05_sd*.json`
  (+ `model_params.npy`).

## Row 2 — high-CV symmetric K=10 (`kaynov2024_instance_13`)

- **Baseline (reproducible):** tuned symmetric echelon base-stock + allocation **gate** →
  **91890.25 ± 99.56**.
- **Policy (replicable, seed-robust):** depth-3 soft tree, `symmetric_echelon_targets` +
  `absolute_augmented`, axis-aligned, linear leaves, warm-started at the gate, σ=0.10,
  pop 32 × 500 gen, batch 16, proportional allocation. 6 seeds {851,852,853,855,856,857}:

  | seed | 851 | 852 | 853 | 855 | 856 | 857 | **mean ± std** |
  |---|---|---|---|---|---|---|---|
  | cost | 84286 | 85741 | 86862 | 84540 | 85547 | 84885 | **85310 ± 946** |
  | vs gate | +8.27% | +6.69% | +5.47% | +8.00% | +6.90% | +7.62% | **+7.16%** |

- **Verdict:** **robustly beats the same-protocol gate by +7.16%** (all 6 seeds beat it; mean−std =
  +6.13% still clears it) — the exploitable structure is dynamic state-dependence under high CV.
  Vs the **PPO scalar 79727.39** (cross-protocol): −7.0% mean, 0/6 below → **robustly below the
  PPO scalar, no PPO beat claimed**.
- **Artifact:** `outputs/owmr_ppo_campaign/i13_seedblock/*.json` + the seed-851 campaign JSON.

## Recommended paper wording (`tab:owmr-results`)
Replace the single-seed het-3 `+1.33%` and high-CV-10 `+6.44%` cells with the 6-seed means:
het-3 **+4.63% ± (gate beat, all seeds positive)**; high-CV-10 **+7.16%**. State the gate beat as
the headline (same-protocol, seed-robust); report PPO as cross-protocol context (het-3 straddles
it within seed noise; high-CV-10 is ~7% short) — explicitly **no PPO beat claimed**, consistent
with the cross-protocol caveat (Kaynov PPO is table-only, env reproduces Kaynov's own rows only to
±1–6%). Report mean ± std over the seeds, not a single seed.

## Reproduce
```bash
# het-3 (instance_12), 6 seeds
for s in 841 842 844 845 846 847; do RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets --policy_state_mode absolute_augmented \
  --leaf_type linear --depth 2 --split_type axis_aligned --temperature 0.10 \
  --warm_start_at_best_base_stock --sigma_init 0.05 --gate_search_paths 64 \
  --training_episodes 800 --es_population 32 --train_seed_batch 24 --holdout_paths 4096 \
  --train_allocation min_shortage --same_seed --seed $s; done
# high-CV-10 (instance_13), 6 seeds: depth 3, symmetric_echelon_targets, sigma 0.10, gen 500, batch 16, proportional
```
