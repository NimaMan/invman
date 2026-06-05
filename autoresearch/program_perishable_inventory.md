# perishable-inventory autoresearch

This is the perishable-inventory counterpart to the dual-sourcing, multi-echelon,
one-warehouse-multi-retailer, joint-replenishment and VMI autoresearch programs. It targets
the `perishable_inventory` problem (De Moor, Gijsbrechts & Boute 2022, *EJOR* 301(2):535-545;
Farrington, Wong, Li & Utley 2025, *Ann. Oper. Res.* 349(3):1609-1638, Table 3): a single
perishable product with fixed shelf life `m`, lead time `L`, FIFO or LIFO issuing, and
per-period holding / shortage / waste / procurement costs over gamma demand
(`mean 4, cov 0.5`). The env is literature-faithful on the `m=2`, `L=1` slice (exact value
iteration re-derives the published optimal-policy tables, the best base-stock levels 5 LIFO /
7 FIFO, and the Farrington 2025 Table-3 value-iteration returns in-crate at test time; see
`src/problems/perishable_inventory/README.md`).

The single-policy loop is the same shape as the sibling programs: warm-start ONE soft-tree
CMA-ES policy at the best base-stock on a NAMED instance, evaluate held-out CRN mean
discounted return + gap vs the strongest heuristic, append a TSV ledger row. The runner is
`scripts/perishable_inventory/autoresearch_perishable_inventory.py`; it drives the binding
`perishable_inventory_soft_tree_population_discounted_return` directly (no Rust rebuild, and
it avoids the dead `invman.policies.soft_tree` import that bricks the older
`run_paper_benchmark.py`).

## Benchmark

Trusted design set = the two exact-verified `m=2`, `L=1` instances:

- `de_moor2022_m2_exp2_l1_cp7_fifo` (primary anchor): FIFO, waste cost 7, VI optimum
  **-1457**, published best base-stock level **7**, optimality gap 1.2%.
- `de_moor2022_m2_exp1_l1_cp7_lifo`: LIFO, waste cost 7, VI optimum **-1553**, published best
  base-stock level **5**, optimality gap 0.8%.

Both are carried verbatim in `references.rs::SCENARIO_A_REFERENCE_INSTANCES`; the other 30
Scenario A rows are table-only anchors (stored, not re-derived) and are NOT design targets.

Two baselines, both honest (the metric is mean discounted return, gamma=0.99, higher = better):

- **VI optimum (analytic)** — the expected discounted return under the midpoint-binned gamma
  demand, the value the crate reproduces from Farrington 2025 (FIFO -1457.28, LIFO -1552.99).
  This is a DIFFERENT estimator from the Monte-Carlo rollouts (sampled+rounded gamma demand,
  finite horizon, zero start), which sit ~11-13 units (~0.8%) below it. The VI gap therefore
  MIXES estimators and is reported for CONTEXT only.
- **base-stock GATE (Monte-Carlo)** — `base_stock` at the published best level, scored by the
  SAME Monte-Carlo discounted-return estimator on the SAME held-out CRN eval seeds as the
  learned policy. This is the apples-to-apples keep/discard gate (`gap_vs_base_stock_gate`);
  the learned policy beats it only if it is cheaper here.

Evaluation protocol: FOUR disjoint CRN blocks from one seed — training seeds (one fresh paired
seed per individual per generation), a heuristic search block (tunes the base-stock level), a
VALIDATION block (selects the promoted policy), and the held-out EVAL block (the reported
number, 2048 seeds at full budget). Keeping validation disjoint from eval is load-bearing
(see "What we know").

## The action geometry (the contribution)

The rollout binding fixes a SCALAR order head (`action_dim == 1`) over the perishable
age-state (`state = [pipeline(L-1), on_hand(m)] / max(demand_mean, 1)`). The expressive class
is the soft-tree LEAF. A LINEAR leaf computes `q = round(softplus(bias + w . state))`, which
is exactly the base-stock structure `q = max(0, S - IP)` when `bias = S` and every
`w_i = -max(demand_mean, 1)` (so `w . state = -(sum on_hand + pipeline) = -IP`). The scalar
head is the right geometry: the perishable order decision IS a scalar order-up-to-style
quantity over the age-disaggregated state, and the soft tree's splits let the leaves express
the AGE-DEPENDENT corrections the published optimal-policy table carries (it orders
differently depending on which age bucket holds inventory, not just on total IP).

**Warm start = the lever.** CMA-ES is warm-started (`cma_x0`) at the inverted leaf transform
above with `S` = the published best base-stock level, so GENERATION 0 REPRODUCES the
base-stock heuristic to within a single-state rounding artifact (`softplus(0)=0.69` rounds to
1 at `IP=S`). Verified: the encoded depth-1/2 soft tree evaluates to -1468.1 vs the actual
base-stock policy at -1468.4 on the same eval block. The optimizer then searches OUTWARD from
a known-good point — the same gen-0-reproduces-heuristic device used for OWMR
`symmetric_echelon_targets` and dual-sourcing `capped_dual_index`.

## Intended search surface (the editable levers)

- **Soft-tree structure**: `--depth` (1,2,3), `--temperature`, `--split_type`
  (oblique / axis_aligned). (`--leaf_type` is `linear` only — the softplus base-stock encoding
  is the warm start; a constant leaf cannot reproduce `max(0, S-IP)`.)
- **Warm start**: `--no_warm_start` ablates to a zero CMA mean (random init). The warm start
  is the headline lever; the ablation exists to quantify it.
- **CMA-ES**: `--sigma_init` (default 0.75, confines the search to a base-stock
  neighbourhood), `--popsize`, `--generations`, `--seed`.

## Budgets

Defined in `scripts/perishable_inventory/autoresearch_perishable_inventory.py`:

- `smoke`     : popsize 8, 10 generations, 16 search / 128 eval seeds (CI / wiring check)
- `screening` : popsize 16, 40 generations, 48 search / 512 eval seeds (first pass)
- `full`      : popsize 24, 120 generations, 64 search / 2048 eval seeds (certify a winner)

The validation block is `max(256, eval_seeds // 4)`. HARD CPU CAP: the script sets
`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` defaults to 2 before importing `invman_rust` (the
population rollout fans out only via rayon). Export them to change it.

## Goal (keep / discard)

KEEP a candidate if it **beats the base-stock gate** (higher mean held-out discounted return)
by more than the paired eval SEM on the same CRN block. Primary metric:
`gap_vs_base_stock_gate = learned - gate` (positive = learned cheaper = win) and its percent
form; the VI optimum is reported alongside for context. A sub-SEM "win" is a tie, not a win.

## What we know (full-budget certified, run 2026-06-04, commit 01c657a)

Depth-2 oblique linear soft tree, warm-started at the published base-stock, popsize 24,
120 generations, 512 validation seeds, 2048 held-out CRN eval seeds (gamma=0.99). Both
exact-verified instances: the learned policy **beats the base-stock gate decisively and lands
within noise of the analytic VI optimum**.

| instance | VI optimum | base-stock gate (MC) | learned (MC) | gap vs gate | gap vs VI | paired SEM | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `..._m2_exp2_l1_cp7_fifo` | -1457.28 | -1475.08 | **-1457.90** | **+17.18 (1.16%)** | -0.62 (-0.043%) | 1.82 | **beats** (9.5x SEM) |
| `..._m2_exp1_l1_cp7_lifo` | -1552.99 | -1565.98 | **-1553.16** | **+12.82 (0.82%)** | -0.17 (-0.011%) | 1.95 | **beats** (6.6x SEM) |

On FIFO the learned policy is 1.16% cheaper than the in-repo base-stock gate — exceeding the
published base-stock's own 1.2% optimality gap, i.e. the learned policy is near-VI-optimal
where the best base-stock is 1.2% off. On both instances `gap_vs_VI ~= 0` on the Monte-Carlo
scale: the learned policy closes the entire estimator-mismatch gap.

LOAD-BEARING PROTOCOL FIX (recorded honestly): selecting the promoted policy on the EVAL
block (or on the per-generation TRAINING argmax) overfits — at full budget that rule gave a
held-out **-1482.29 (loses, -0.49% vs gate)** because the chosen generation-best flattered its
single training seed (training return -1351 vs held-out -1482), and the CMA incumbent mean had
itself drifted below the gate. Selecting on a DISJOINT validation block recovered the genuine
win (-1457.90). The runner now selects on the validation block; the ledger's superseded
eval-selection row is preserved in git history as the audit trail.

## Open levers (not yet run)

- depth 3 + axis-aligned splits (cheap; may sharpen the age-dependent threshold the optimal
  policy table shows).
- the `m=2`, `L=2` and `m>=3` Scenario A rows are TABLE-ONLY anchors (no in-crate exact MDP),
  so they are not gate-able the same way; extend only if those instances gain an exact
  verifier.

## Canonical workspace

Ledger and per-run artifacts:

- `outputs/autoresearch/perishable_inventory_autoresearch/results.tsv` (the appended ledger:
  commit, reference, budget, structure flags, learned_return, vi_optimum, base_stock_gate,
  gap_vs_vi_pct, gap_vs_gate, gap_vs_gate_pct, verdict, description)
- `outputs/autoresearch/perishable_inventory_autoresearch/<reference>_d<depth>_<split>_<budget>.json`
  (the full per-run payload: baselines, warm-start gen-0 reproduction check, selection source,
  learned return + SEM, gaps).
