# OWMR depth-3 AutoResearch campaign + seed-robustness audit (2026-06-06)

**Objective.** Use autoresearch (CMA-ES soft-tree policy search) to push the
one-warehouse multi-retailer learned policy past the −1.82% / −5.86% plateau the prior
three sessions reached on instances 12 / 13 — and to report results that are
**robust, not seed-dependent**.

## Headline (seed-robust)

On `kaynov2024_instance_12`, a **gentle depth-3** soft tree robustly **beats the tuned
in-repo base-stock + allocation gate by ≈ +5.0%** (mean over 6 optimizer seeds, every
seed positive, +4.21%…+7.17%) — roughly **doubling** the prior depth-2 gate-beat margin
(+2.59%). This is the clean, same-protocol claim.

It does **not** robustly beat the published Kaynov PPO **scalar**. Across a fixed
configuration the seed-to-seed spread (~±7 cost units) is *larger* than the ~10.6-unit
margin to PPO, so individual seeds **straddle** the PPO line. A single seed (841) lands
+0.94% below PPO and verifies across disjoint held-out blocks, but reporting that as a
"PPO beat" would be a seed cherry-pick. The honest statement is: **the depth-3 policy
closes the PPO gap from −1.82% to within seed noise of the PPO scalar (mean +0.69%,
range −0.13%…+1.36% at σ=0.05), while robustly beating the same-protocol gate.**

> **Cross-protocol caveat (from `literature/README.md`).** Kaynov's PPO row is
> table-only; this env reproduces Kaynov's *own* base-stock rows only within ~1–6% with a
> regime-dependent sign, under an unverified demand convention. So even a clean
> below-scalar number would be *cross-protocol context*, not a head-to-head beat — the
> same status the paper already assigns the general-network PPO figure. The defensible
> claim is the same-protocol gate beat.

## Seed-robustness (the load-bearing finding)

Config: `echelon_targets_with_alloc_targets / absolute_augmented / depth-3 / axis_aligned
/ linear / t0.10`, warm-started at the gate, pop 32 × 800 gen, train_seed_batch 24
(`--same_seed` CRN), 4096 held-out paths, deployed = better of {trained xbest, gate}.

| σ | seeds | learned mean ± std | vs gate (mean / min) | vs PPO scalar (mean / range) | seeds below PPO |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.03 | {846} | 1085.73 (N=1) | +7.17% | +2.97% | 1/1 |
| 0.05 | {841,842,844,845} | **1111.23 ± 7.06** | **+4.99% / +4.21%** | +0.69% / −0.13%…+1.36% | 3/4 |
| 0.07 | {847} | 1118.99 (N=1) | +4.33% | −0.01% | 0/1 |

Per-seed (σ=0.05): 1103.72 / 1108.35 / 1112.51 / 1120.36. The gate beat is tight and
always positive; the PPO-scalar comparison is dominated by optimizer-seed noise. **Rule
adopted for the paper: never report a single-seed headline. Report the multi-seed mean ±
std and make claims only where the aggregate clears the comparator beyond that spread.**

The σ=0.03 single point (1085.73) is the best observed and hints lower sigma may be
robustly better, but N=1 — it needs its own multi-seed block before any claim.

## Disjoint-block check on the seed-841 policy (generalization, not a headline)

`verify_ppo_beat_disjoint_blocks.py` re-scores the seed-841 trained policy on CRN blocks
never used in training/gate-search/deployment selection. It confirms that *that specific
trained policy* generalizes (it is not overfit to the deployment block) — but it says
nothing about seed-robustness, which the table above addresses.

| Block seed | proportional | min_shortage |
| ---: | ---: | ---: |
| 900 000 (orig) | 1108.35 ± 2.06 | 1108.42 ± 2.06 |
| 2 000 000 | 1107.83 ± 2.08 | 1107.89 ± 2.08 |
| 3 000 000 | 1110.03 ± 2.09 | 1110.15 ± 2.09 |
| 4 000 000 | 1110.19 ± 2.05 | 1110.22 ± 2.05 |

## instance_13 (K=10 symmetric high-CV)

Best config `i13_E1_sym_d3_axis_s0p10` (depth-3 `symmetric_echelon_targets` +
`absolute_augmented`): 84286, gate beat **+8.27%** (prior +6.44%), still −5.72% vs the
PPO scalar. Single seed so far — needs a multi-seed block before any aggregate claim. The
deeper symmetric tree extends the gate-beat; the PPO-scalar gap is structurally larger and
the N(5,14) demand-convention ambiguity makes that scalar the least trustworthy of the
suite.

## What changed vs the prior plateau (preliminary; under agent review)

Prior sessions saturated at 1139.34 (−1.82%) with depth-2 axis-aligned local restart
chains; richer/larger searches overfit the small training batch and fell back to the gate.
This campaign moved the frontier with two untried levers, but the **load-bearing one is
not yet isolated** (depth-3 vs absolute_augmented vs more training paths vs gentle sigma
are confounded in the winning config). An ablation is part of the spawned review.

## Reproduce

```bash
# multi-seed depth-3 block (report mean ± std, NOT a single seed)
for s in 841 842 844 845; do RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets --policy_state_mode absolute_augmented \
  --leaf_type linear --depth 3 --split_type axis_aligned --temperature 0.10 \
  --warm_start_at_best_base_stock --sigma_init 0.05 \
  --gate_search_paths 64 --training_episodes 800 --es_population 32 --train_seed_batch 24 \
  --holdout_paths 4096 --train_allocation min_shortage --same_seed --seed $s; done

python scripts/one_warehouse_multi_retailer/run_owmr_ppo_campaign.py --only both --max_parallel 8
```

Artifacts: `outputs/owmr_ppo_campaign/` (gitignored): per-config JSON + logs,
`campaign_results_both.tsv`, `repro_d3/`.
