# Literature Overview

## Verification status: LITERATURE-VERIFIED (exact m=2/L=1 slice only)

Honest scope statement (both cited papers were independently confirmed REAL and
metadata-corrected during a librarian audit, 2026-05; see "Primary sources"):

- LITERATURE-VERIFIED: the four `m = 2`, lead-time-1 instances (121 states each).
  The repo's exact value-iteration MDP (`value_iteration_mdp.rs`) is RE-DERIVED by
  the in-repo solver at test time (not merely compared against stored numbers) and
  reproduces THREE independent published quantities cell-for-cell, asserted in
  `tests/verification.rs`.
- TABLE-ONLY (NOT independently re-derived): the remaining 28 Scenario A rows
  (1331 to ~1.77M states). Their Farrington Table 3 numbers are stored as
  documented anchors only; the in-repo exact summary caps at 2000 states
  (`bindings.rs`). Do not read these 28 rows as "verified".

The three re-derived published quantities for the verified slice are:

1. De Moor et al. (2022) optimal-policy tables (9x9), both LIFO (exp 1) and FIFO
   (exp 2) — `matches_published_policy_table = True`. (Repo labels these "Figure 3";
   the exact published figure number was not independently confirmed, see below.)
2. De Moor et al. (2022) best base-stock levels: `5` (LIFO), `7` (FIFO)
   — `matches_published_base_stock_level = True`.
3. Farrington et al. (2025) Table 3 value-iteration mean returns: `-1553` (LIFO),
   `-1457` (FIFO), reproduced exactly to the rounded integer
   — `matches_published_value_iteration_mean_return = True`. Confirmed present in
   the published Table 3 as -1553±61 (LIFO) and -1457±59 (FIFO) (PMC copy).

The FlowNet policy-performance verifier additionally reproduces the published
best base-stock return for the FIFO instance: published `-1474`, reproduced
`-1475` (within tolerance 1.0; the 1-unit residual is rounding noise in the
discounted-return integration).

Evidence (run on the installed `invman_rust`):
`perishable_inventory_exact_mdp_summary("de_moor2022_m2_exp2_l1_cp7_fifo")` and
`...("de_moor2022_m2_exp1_l1_cp7_lifo")` return all three `matches_published_*`
flags `True`; `perishable_inventory_flownet_policy_verification_summary()` returns
`all_observed_targets_within_tolerance = True`. The executable assertions live in
`tests/verification.rs`. References are encoded verbatim in `references.rs`.

This is a genuine multi-anchor literature reproduction, not a self-consistency
check against the repo's own solver.

## Primary sources

- De Moor, Bram J.; Gijsbrechts, Joren; Boute, Robert N. (2022). "Reward shaping
  to improve the performance of deep reinforcement learning in perishable
  inventory management". European Journal of Operational Research, 301(2),
  535-545.
  - DOI: https://doi.org/10.1016/j.ejor.2021.10.045
  - Provides the Scenario A settings, the m=2 experiment 1/2 optimal-policy tables,
    and the best base-stock levels (5 LIFO, 7 FIFO). Verified real and metadata-
    correct via Crossref/RePEc (https://ideas.repec.org/a/eee/ejores/v301y2022i2p535-545.html).
    The exact published figure NUMBER (the repo labels these tables "Figure 3")
    was not independently confirmed by the librarian audit because the EJOR full
    text is paywalled; Farrington et al. (2025) confirm De Moor et al. "fully
    specified the optimal and heuristic policies for two experiments in their
    paper", which is the substance the repo relies on.
- Farrington, Joseph; Wong, Wai Keong; Li, Kezhi; Utley, Martin (2025). "Going
  faster to see further: graphics processing unit-accelerated value iteration and
  simulation for perishable inventory control using JAX". Annals of Operations
  Research, 349(3), 1609-1638.
  - DOI: https://doi.org/10.1007/s10479-025-06551-6
  - Open-access copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC12350524/ ;
    arXiv preprint: https://arxiv.org/abs/2303.10672
  - Table 3 reports value-iteration and simulation-optimization mean returns
    (±std) for all 32 Scenario A settings from De Moor et al. (2022). The m=2,
    lead-time-1 value-iteration returns -1553±61 (LIFO) and -1457±59 (FIFO) are
    confirmed in the published Table 3 (PMC copy).
  - CITATION NOTE: an earlier version of this file and references.rs cited this as
    "Farrington, Li, Utomo, et al." with only a PMC URL. That author list was
    wrong: there is no author "Utomo" (a corruption of "Utley"), and co-author
    "Wong" was omitted. The correct author order is Farrington, Wong, Li, Utley.

## Reference set fidelity (all 32 rows checked against Farrington Table 3)

`references.rs` carries all 32 Scenario A settings (lifetime m in {2,3,4,5},
8 experiments each, lead time 1 or 2, FIFO/LIFO, waste cost 7 or 10). Spot-checks
of the transcription against Farrington et al. (2025) Table 3 (m=2 exp 1/2/5/6/7/8,
m=4 exp 6/8, m=5 exp 6/8) all match verbatim, INCLUDING the published duplicates
(m=4 and m=5 experiments 6 and 8 all report `-1432 / -1453` in the paper itself).
The duplication is in the published source, not a repo transcription error.

Only the four `m = 2`, lead-time-1 instances (121 states) are small enough for the
exact summary, which caps at 2000 states (`bindings.rs:296`). The remaining 28
instances (1331 to 1.77M states) carry the published Farrington numbers as
documented anchors but are NOT independently re-derived by the in-repo solver.

## Canonical instance roles

- exact verification + paper exact slice:
  - `de_moor2022_m2_exp1_l1_cp7_lifo`
  - `de_moor2022_m2_exp2_l1_cp7_fifo` (primary)
- practical benchmark instance:
  - `de_moor2022_m4_exp6_l2_cp7_fifo`

## Benchmark policies carried by the repo

- `base_stock` (single base-stock level S)
- `bsp_low_ew` (low-inventory / estimated-waste base-stock with a threshold)
- `soft_tree` (the repo's learned structured policy, CMA-ES optimized)

(De Moor's own DQN / shaped-DQN comparators from the paper are documented as the
paper's benchmark policies but are not re-implemented here; the repo's learned
comparator is the soft tree.)

## Benchmark results (exact slice)

See `../experiments/reports/exact_slice_report.md` (refresh with
`scripts/perishable_inventory/run_exact_slice_benchmark.py`). Summary on the
shared Monte-Carlo estimator and eval seeds:

- FIFO `m = 2`: CMA-ES depth-2 soft tree beats the best tuned heuristic by ~12-16
  discounted-return units (3-4 SEM) and is statistically indistinguishable from
  the optimum on the Monte-Carlo scale.
- LIFO `m = 2`: the `sigmoid_linear`-leaf soft tree beats the best heuristic by
  ~14 units; the `linear`-leaf tree landed in a worse basin (honest negative).
  LIFO is near heuristic-optimal (exact-vs-heuristic gap ~5), matching De Moor.

### Estimator caveat (important for reading the gaps)

Two distinct discounted-return estimators coexist:

- `exact_value_iteration` is the ANALYTIC expected discounted return under the
  midpoint-binned gamma demand (`value_iteration_mdp.rs`, burn-in 100 + eval 365,
  gamma 0.99). This is the value matched to Farrington Table 3.
- The heuristic and soft-tree rows are MONTE-CARLO means over sampled-and-rounded
  gamma demand rollouts (`heuristics::policy_discounted_return`,
  `rollout::rollout_discounted_return`).

On the same FIFO instance the optimal base-stock level (S=7) evaluates to ~-1468
to -1473 under the Monte-Carlo estimator versus the analytic -1457 — a systematic
~1% offset (≈6 SEM at 512 seeds) that is a property of the estimator, not of the
policy. Compare policies WITHIN one estimator: the `gap_to_best_heuristic` column
is apples-to-apples; the `gap_to_exact_optimum` column mixes estimators and is
informational only.

## Remaining steps

- A same-estimator optimal reference (the full exact-optimal POLICY rolled out on
  the Monte-Carlo eval seeds) would remove the estimator caveat from the gap-to-
  optimum column. There is no current binding to roll out the tabular optimal
  policy; adding one is a small Rust addition (see blockers) and is deferred.
- Independent re-derivation of the 28 larger Farrington rows would require raising
  the 2000-state cap in `bindings.rs` and is memory-bound for m=5/L=2 (1.77M
  states); deferred.
- The deprecated `run_paper_benchmark.py` / `common.py` import path
  (`invman.policies.soft_tree`) is repo-wide drift; the new
  `run_exact_slice_benchmark.py` is the working replacement.
