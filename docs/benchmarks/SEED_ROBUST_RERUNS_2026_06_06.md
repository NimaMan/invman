# Seed-Robust Re-Runs — 2026-06-06

Re-running the at-risk single-seed / best-of-N "beats X" claims as **≥5 independent
optimizer (CMA-ES) seed** blocks, per the project mandate (`MEMORY:
seed-robust-reporting-standard`). Every headline is reported as a **seed-MEAN ± cross-seed
STD** vs the **same-protocol gate** (the env's own grid-searched best constant base-stock /
the strongest in-env heuristic). Cross-protocol comparators (A3C, PPO, DRL) are **CONTEXT
ONLY** — never head-to-head beats. A beat is claimed ONLY if the seed-mean margin exceeds the
cross-seed std (ideally every seed on the winning side).

All runs: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, ≤ ~12 cores total, reusing the EXISTING
per-problem runners under `scripts/<problem>/` (no new envs). Cross-seed std (over independent
optimizer seeds) is sharply separated from within-policy held-out CRN eval SEM (the noise of a
single policy); the verdict is driven by the cross-seed std.

---

## Top summary table

| Claim | Old (single-seed / best-of-N) | New N-seed mean ± std (vs same-protocol gate) | Verdict |
|---|---|---|---|
| **multi_echelon divergent setting1** | best-of-N: learned 779.81 vs gate 911.39 → **−14.44%** | N=5: **−14.74% ± 1.60%** (5/5 below gate) | **ROBUST BEAT vs gate** |
| **multi_echelon divergent setting2** | best-of-N: learned 973.55 vs gate 1137.79 → **−14.43%** | N=5: **−12.04% ± 2.26%** (5/5 below gate) | **ROBUST BEAT vs gate** |
| **gbk set1 (CardBoard)** | best-of-N (N=2): −22.4% / −26.7% vs gate 10354.8 | N=5: **−24.31% ± 1.83%** (5/5 below gate) | **ROBUST BEAT vs gate** |
| **gbk Kunnumkal-Topaloglu** | best-of-N (N=2): ~−37% vs gate 3930.4 | N=5: **−36.79% ± 0.29%** (5/5 below gate) | **ROBUST BEAT vs gate** |
| **dual_sourcing "beats CDI on 2 rows"** | single-seed: −0.009% / −0.041% vs CDI | N=5: all 6 rows **+0.09%…+0.96% ABOVE CDI** (best 2 rows PARITY at +0.09% ± 0.13%) | **COLLAPSES — parity at best, no beat** |
| **joint_pricing_inventory "+25.15% profit"** | single-seed: 216.060 vs heuristic 171.513 (+25.15%) | N=5: **+28.76% ± 0.69%** profit (5/5 beat gate) | **ROBUST BEAT vs gate** |
| **random_yield (MED, N=4)** | prior N=4: +4.25% ± 1.92% vs LIR | N=4 (5th seed bricked): **+4.25% ± 1.92%** (4/4 beat) | **provisional beat; DEFERRED to N=5** |

(Cross-protocol context, NOT head-to-head: divergent A3C savings 8.95% / 12.09%; gbk PPO best
8714 / DRL 3724. These are different protocols/algorithms; the repo implements no A3C/PPO, so
they are reported alongside as context, never as a beat the learned policy "wins".)

---

## HIGH-3 — general_backorder_fixed_cost: set1 + Kunnumkal-Topaloglu divergent (N=2 → N=5)

**Runner:** `scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py`
(reused verbatim; `--reference <name> --budget full --seed <s>`).
**Budget:** `full` (popsize 24, 80 generations, 12 train seeds, **2000 held-out CRN eval
seeds**, sigma 0.20, depth-2 oblique constant leaf, warm-started at the published levels so
gen-0 reproduces the benchmark). No budget reduction.
**Same-protocol gate:** the repo node-base-stock simulation at the published levels (set1
`[82,100,64,83,35,35,35,35,35]`; KT `[124,30,30,30]`), 500/1000 reps × 3 seeds.
**Cross-protocol context (NOT a beat):** PPO best 8714 (set1) / DRL 3724 (KT).

### set1 (`geevers2023_general_set1`, CardBoard 4-warehouse / 5-retailer, Poisson(15))

Gate (repo reproduction) = **10354.8** (published benchmark 10467; gap −1.1%).

| seed | learned held-out mean | gap vs gate |
|---|---|---|
| 123 (existing) | 8034.8 | −22.41% |
| 777 (existing) | 7590.7 | −26.69% |
| 9101 (new) | 7685.9 | −25.77% |
| 9102 (new) | 7909.8 | −23.61% |
| 9103 (new) | 7963.6 | −23.09% |

- **Learned seed-mean = 7837.0 ± 189.7** (cross-seed std).
- **Savings vs gate = −24.31% ± 1.83%; 5/5 seeds below the gate.**
- Margin (24.31%) ≫ cross-seed std (1.83%) → **ROBUST BEAT vs the same-protocol gate.**
- All 5 seeds also fall below the published PPO best 8714 (cross-protocol context only).

### Kunnumkal-Topaloglu divergent (`kunnumkal_topaloglu_divergent`, 1→1→3, resampled Poisson)

Gate (repo reproduction) = **3930.4** (published benchmark 4059; gap −3.2%).

| seed | learned held-out mean | gap vs gate |
|---|---|---|
| 123 (existing) | 2469.1 | −37.18% |
| 777 (existing) | 2477.9 | −36.96% |
| 9101 (new) | 2488.4 | −36.69% |
| 9102 (new) | 2498.4 | −36.43% |
| 9103 (new) | 2489.0 | −36.67% |

- **Learned seed-mean = 2484.6 ± 11.3** (cross-seed std).
- **Savings vs gate = −36.79% ± 0.29%; 5/5 seeds below the gate.**
- Margin (36.79%) ≫ cross-seed std (0.29%) → **ROBUST BEAT vs the same-protocol gate.**
- All 5 seeds also fall below the published DRL 3724 (cross-protocol context only).

**VERDICT (both):** ROBUST BEAT. The huge margins (>22%, >36%) dwarf the cross-seed std; the
best-of-N N=2 framing understated robustness — the full 5-seed spread stays well inside a
robust beat.

**Recommended paper-table wording:**
> On the general-backorder fixed-cost set-1 (CardBoard) network the learned state-dependent
> node-base-stock policy reduces cost vs the reproduced constant node-base-stock gate (10354.8)
> by **24.3% ± 1.8% over 5 independent CMA-ES seeds** (all 5 seeds below the gate). On the
> Kunnumkal-Topaloglu divergent instance the reduction is **36.8% ± 0.3% over 5 seeds** vs the
> reproduced gate (3930.4), all 5 below. (For context, all seeds also fall below the published
> PPO best 8714 / DRL 3724 — a different algorithm and training protocol, reported as context,
> not a head-to-head beat.)

---
## HIGH-1 — multi_echelon divergent special-delivery (Gijs setting1 & setting2): best-of-N → N=5

**Runner:** `scripts/multi_echelon/seed_robust_divergent_multi_echelon.py` (a thin ≥5-seed
driver that reuses `train_multi_echelon_policy.py`'s `train_one`, `published_a3c_savings`, and
`best_constant_base_stock_over_operating_region` VERBATIM — no new env).
**Budget:** `full` (training_episodes 400, ES population 24, train horizon 3000, **eval horizon
30000 × 6 eval seeds**, sigma 2.0, temp 0.25). Action design `direct_level`, depth sweep {2,3}
(best per seed taken — the SAME sweep as the original claim). No budget reduction.
**Same-protocol gate:** the env's own in-region best constant base-stock (grid search
`yw∈{0..700:25}`, `yr∈{0..60:5}` re-scored on the eval protocol). The gate grid search is
itself seeded, so the gate moves marginally per seed (≈±0.5); each learned cost is compared to
the gate AT THE SAME SEED (paired).
**Cross-protocol CONTEXT (NOT a beat):** Gijsbrechts (2022) published A3C savings 8.95%
(setting1) / 12.09% (setting2). The repo implements no A3C; these are reported alongside as
context only, never as a head-to-head beat.

### setting1 (`gijsbrechts2022_setting1`, gijs_2022 mode, μ=5, K=10, lw=2 lr=2)

| seed | gate (yw=300,yr=25) | best learned (design) | savings vs gate |
|---|---|---|---|
| 9001 | 910.49 | 766.02 (direct_level d3) | +15.87% |
| 9002 | 909.66 | 800.65 (direct_level d2) | +11.98% |
| 9003 | 910.07 | 766.01 (direct_level d3) | +15.83% |
| 9004 | 911.02 | 774.43 (direct_level d3) | +14.99% |
| 9005 | 910.46 | 773.63 (direct_level d3) | +15.03% |

- Gate seed-mean **910.34 ± 0.51**; learned seed-mean **776.15 ± 14.27**.
- **Savings vs gate = +14.74% ± 1.60% (5/5 seeds below gate).** Margin (14.74%) ≫ cross-seed
  std (1.60%) → **ROBUST BEAT vs the same-protocol gate.**
- A3C context only: 8.95% (different protocol; not a head-to-head beat).

### setting2 (`gijsbrechts2022_setting2`, gijs_2022 mode, μ=0, K=10, lw=5 lr=3)

| seed | gate (yw=525,yr=25) | best learned (design) | savings vs gate |
|---|---|---|---|
| 9001 | 1137.95 | 1028.26 (direct_level d3) | +9.64% |
| 9002 | 1137.54 | 990.96 (direct_level d3) | +12.89% |
| 9003 | 1138.18 | 1028.69 (direct_level d2) | +9.62% |
| 9004 | 1137.83 | 982.37 (direct_level d2) | +13.66% |
| 9005 | 1138.69 | 975.09 (direct_level d2) | +14.37% |

- Gate seed-mean **1138.04 ± 0.43**; learned seed-mean **1001.07 ± 25.63**.
- **Savings vs gate = +12.04% ± 2.26% (5/5 seeds below gate).** Margin (12.04%) ≫ cross-seed
  std (2.26%) → **ROBUST BEAT vs the same-protocol gate.**
- A3C context only: 12.09% (different protocol; not a head-to-head beat).

**VERDICT (both settings):** ROBUST BEAT vs the same-protocol best-constant-base-stock gate.
The best-of-N headlines (−14.44% / −14.43%) survive as robust beats; the honest 5-seed means
are +14.74% ± 1.60% (setting1) and +12.04% ± 2.26% (setting2). NOTE setting2's seed-mean
(12.04%) is a hair below the prior best-of-N (14.43%): the best-of-N over-stated the typical
seed by ~2.4pp, but the beat is still robust (every seed positive, margin ≫ std).

**Recommended paper-table wording:**
> On the divergent special-delivery instances (Gijsbrechts 2022 settings 1 & 2) the learned
> direct-level soft-tree policy reduces cost vs the env's own grid-searched best constant
> base-stock gate by **14.7% ± 1.6% (setting 1)** and **12.0% ± 2.3% (setting 2)** over 5
> independent CMA-ES seeds, with all 5 seeds below the gate in both cases. (For context, the
> published A3C savings on these instances are 8.95% / 12.09%; A3C is a different algorithm and
> training protocol that the repo does not re-run, so this is reported as context, not a
> head-to-head beat.)

---
## MED — joint_pricing_inventory: "+25.15% profit" single-seed → N=5

**Runner:** `scripts/joint_pricing_inventory/train_soft_tree_reference.py` (reused verbatim;
`--depth 2 --leaf_type linear --training_episodes 400 --es_population 16 --train_seed_batch 8
--eval_seeds 4096 --seed <s>`). No budget reduction (4096 held-out eval seeds, the paper eval
size).
**Same-protocol gate:** the best benchmark heuristic on the primary 18-period Poisson instance
= `static_price_base_stock` (cost −171.958; profit 171.958). This is a DETERMINISTIC re-eval on
the CRN block, so the gate is fixed across optimizer seeds; only the learned policy varies.
Costs are negative (profit maximization); a "beat" = lower (more negative) cost = higher profit.

| seed | learned cost (profit) | static_price_base_stock cost | profit improvement % |
|---|---|---|---|
| 9001 | −219.356 | −171.958 | +27.56% |
| 9002 | −221.855 | −171.958 | +29.02% |
| 9003 | −221.589 | −171.958 | +28.86% |
| 9004 | −222.399 | −171.958 | +29.33% |
| 9005 | −221.886 | −171.958 | +29.03% |

- Learned cost seed-mean **−221.417 ± 1.189** (cross-seed std).
- **Profit improvement vs the best heuristic gate = +28.76% ± 0.69% (5/5 seeds beat).**
- Margin (28.76%) ≫ cross-seed std (0.69%) → **ROBUST BEAT vs the same-protocol gate.**
- The prior single-seed +25.15% (216.060 vs 171.513) is confirmed robust; the small numeric
  difference from the prior gate (171.958 here vs 171.513) is an eval-protocol detail, not a
  change in verdict.

**Recommended paper-table wording:**
> On the primary joint-pricing-and-inventory instance the learned price-reactive soft-tree
> policy improves profit over the best benchmark heuristic (static-price base-stock) by
> **28.8% ± 0.7% over 5 independent CMA-ES seeds** (all 5 seeds beat the heuristic), on a
> 4096-seed held-out CRN block.

---
## HIGH-2 — dual_sourcing "beats CDI on 2 of 6 rows" single-seed → N=5

**Runner:** `scripts/dual_sourcing/benchmark_full_suite.py` (reused verbatim; one full-suite run
per optimizer seed, run_tag `ds_seedrobust_s<seed>`, `--only` the SELECTED spec
`soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets` — the spec the original claim
used). Aggregated by `scripts/dual_sourcing/aggregate_seed_robust_cdi.py`.
**Budget:** `full` (training_episodes 1500, ES population categorical {32,64,96,128}, train
horizon 2000, eval horizon 10000 × 3 eval seeds, sigma 3.0). No budget reduction; restricted to
the SELECTED spec only so 5 seeds × 6 rows fit the CPU/time cap.
**Same-protocol gate / comparator:** the best Gijsbrechts heuristic on each row, which is
`capped_dual_index (CDI)` on all 6 rows (the ~0% published-optimality-gap proxy). CDI is a
deterministic grid-searched gate (fixed across optimizer seeds); only the learned policy varies.
Seeds: 9001..9005.

| row | learned seed-mean | CDI (gate) | gap% vs CDI (mean ± std) | seeds beating CDI | verdict |
|---|---|---|---|---|---|
| dual_l2_ce105 | 217.136 | 216.244 | **+0.413% ± 0.384%** | 0/5 | robustly above CDI |
| dual_l2_ce110 | 221.393 | 219.289 | **+0.960% ± 0.840%** | 0/5 | robustly above CDI |
| dual_l3_ce105 | 216.609 | 216.398 | **+0.098% ± 0.129%** | 1/5 | PARITY (within seed noise) |
| dual_l3_ce110 | 221.592 | 220.027 | **+0.711% ± 0.136%** | 0/5 | robustly above CDI |
| dual_l4_ce105 | 216.608 | 216.418 | **+0.088% ± 0.131%** | 1/5 | PARITY (within seed noise) |
| dual_l4_ce110 | 221.554 | 220.409 | **+0.519% ± 0.168%** | 0/5 | robustly above CDI |

**VERDICT: the "beats CDI on 2 rows" claim COLLAPSES TO PARITY (at best).** No row has a
negative seed-mean gap; every row's seed-mean is ABOVE CDI. The two rows that the single-seed
claim flagged as beats (dual_l2_ce110 −0.009%, dual_l4_ce110 −0.041%) are, over 5 seeds,
robustly ABOVE CDI (+0.96%, +0.52%; 0/5 seeds below). The closest the learned policy gets is
PARITY on the two tightest rows (dual_l3_ce105 +0.098% ± 0.129%, dual_l4_ce105 +0.088% ± 0.131%,
each with 1/5 seeds dipping just below CDI). This is exactly the repo-memory expectation
("matches CDI, margin < CDI's optimality band"). The single-seed −0.009%/−0.041% was a
best-of-the-noise straddle inside CDI's <0.11% optimality band, not a robust beat.

**Recommended paper-table wording:**
> Across the six Gijsbrechts Figure-9 dual-sourcing rows the learned soft-tree policy MATCHES
> the capped-dual-index optimal proxy: the seed-mean relative gap vs CDI over 5 independent
> CMA-ES seeds ranges from **+0.09% ± 0.13%** (the tightest rows, dual_l3_ce105 / dual_l4_ce105)
> to **+0.96% ± 0.84%** (dual_l2_ce110), every row's seed-mean lying at or just above CDI. No row
> robustly beats CDI; the earlier single-seed "beats CDI on 2 rows" was a best-of-noise straddle
> inside CDI's <0.11% optimality band. Headline: **the learned policy clears the published A3C
> gaps (0.51–1.85%) by matching CDI**, not by beating it.

---

## MED — random_yield_inventory: existing N=4 evidence (5th seed DEFERRED, runner bricked)

**Status: N=4 only; could not reach ≥5 cleanly.** The existing seed-robust runs use
`scripts/random_yield_inventory/train_soft_tree_reference.py` (d1, linear, 800 episodes,
train_seed_batch 8) over seeds {123, 456, 789, 2026}. Attempting a 5th seed (555) reproduces a
known brick: training completes but the report-writing step crashes with
`'Policy' object has no attribute 'action_spec'` (line ~209), so a comparable JSON is not
emitted. The 4 committed seeds were produced by an older code path.

The existing N=4 picture (LIR = linear_inflation gate; soft tree vs LIR, discounted cost):

| seed | soft tree | LIR gate | savings vs LIR |
|---|---|---|---|
| 123 | 191.923 | 205.237 | +6.49% |
| 456 | 197.380 | 204.983 | +3.71% |
| 789 | 201.320 | 205.280 | +1.93% |
| 2026 | 196.311 | 206.400 | +4.89% |

- soft tree 4-seed mean **196.73 ± 3.86**; LIR 4-seed mean 205.47 ± 0.63.
- Savings vs LIR = **+4.25% ± 1.92% (4/4 beat)** — margin > std at N=4, so it LOOKS robust, but
  it is N=4, one short of the ≥5 mandate, and the runner cannot currently produce a 5th
  comparable seed. **VERDICT: provisional robust beat at N=4; needs the report-path fix
  (`action_spec`) to confirm at N=5.** DEFERRED.

---

## Deferred MED items (honest)

| Item | Why deferred |
|---|---|
| **perishable_inventory ×5** | The seed-robust path needs `autoresearch_perishable_inventory.py` per-instance (FIFO/LIFO, selection-split-sensitive) over 5 seeds × 5 cells; the companion `run_paper_benchmark.py`/`common.py` are bricked (import deleted `invman.policies.soft_tree`). Not enough time to wire and validate a clean 5-seed sweep without risking dubious numbers. |
| **joint_replenishment 6/16 MOQ-beats** | `autoresearch_joint_replenishment.py` imports cleanly but the claim is a 16-setting surface (single-seed); a faithful 5-seed re-run of the 6 winning settings (esp. the best-of-N setting-10 flip) is a multi-run job not reached in this session. |
| **nonstationary beats-DP 8/8** | `run_practical_benchmark.py` has no `--seed` and trains no policy (heuristics on one fixed path); the learned-policy multi-seed training path was not located/validated in time. |
| **random_yield 5th seed** | Runner report-path bricked (see above); N=4 provisional result recorded. |

These are all MED ("if time") items; the four HIGH/MED claims that WERE re-run (divergent ×2,
gbk ×2, dual_sourcing, joint_pricing) are the mandated priority and are complete.
