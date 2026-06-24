# Perishable inventory — benchmark card

**One-line MDP:** state = age-structured on-hand stock (one bucket per remaining shelf-life period) plus the in-transit pipeline; action = scalar order quantity `q_t ∈ {0..q̄}`; one-period cost = `c·q_t + h·Σ surviving on-hand + p·lost + c_w·outdated`; objective = maximize expected discounted **negated** cost (γ=0.99, higher = less cost) over a post-warm-up window.

**Status:** verified_rerun (genuine value-iteration re-derivation of 3 independent published quantities on the four `m=2`, `L=1`, 121-state instances; the other 28 Scenario A rows are TABLE-ONLY anchors).  **Paper:** §Perishable inventory (`\label{sec:perishable}`, ~line 1769) of `paper/learning_inventory_control_policies_es.tex`.

## Problem formulation

Single item, periodic review, fixed shelf life `m` periods, deterministic lead time `L≥1`. Demand `D_t` is i.i.d., sampled from a Gamma(`μ`, `cv`) distribution and rounded to the nearest non-negative integer. Unmet demand is **lost** (lost-sales, not backorder). Model of De Moor, Gijsbrechts & Boute (2022, EJOR 301(2):535-545), re-derived/benchmarked by Farrington, Wong, Li & Utley (2025, Ann. Oper. Res. 349(3):1609-1638, Table 3).

- **Timing of a period** (env.rs `step_state`): observe age-structured stock → place order `q_t` → demand `D_t` realized and served from on-hand under the issuing rule → unmet demand lost → **oldest leftover bucket outdates (waste)** → survivors age one bucket → the order placed `L` periods earlier arrives into the **youngest** bucket. With `L=1`, `q_t` arrives at the start of `t+1` and there is no in-transit pipeline.
- **State:** `S_t = (q_{t-L+1},…,q_{t-1}; o¹_t,…,o^m_t)` = pipeline (`L-1` entries) followed by on-hand by age (`o¹` youngest, `o^m` oldest). For the verified slice `m=2, L=1` this collapses to the 2-D age vector `(o¹_t, o²_t)`. Observation layout (env.rs `build_raw_state`): pipeline entries first, then on-hand buckets; no silent normalization in the env (any scale-by-mean is done in the policy/rollout).
- **Action:** scalar order `a_t = q_t ∈ {0,…,q̄}`, `q̄` the per-period order cap (10 for the reference instances).
- **Transition / issuing:** demand consumes `min(D_t, O_t)` of total on-hand `O_t`; FIFO issues oldest-first (env.rs serves from highest age index down), LIFO youngest-first. Leftover in the oldest bucket `r^m_t` is the waste `w_t`; lost `ℓ_t = (D_t - O_t)^+`.
- **One-period cost** (Eq. `eq:perish-cost`): `c_t = c·q_t + h·Σ_{j=1}^{m-1} r^j_t + p·ℓ_t + c_w·w_t` (procurement on the order, holding on surviving non-expiring on-hand, lost-sales penalty, outdating/waste cost).
- **Long-run objective** (Eq. `eq:perish-objective`): `G(θ) = E[ Σ_{t=W+1}^{T} γ^{t-W-1} (−c_t) ]`, maximized. Reported returns are **negative** (higher = cheaper). Protocol: `γ=0.99`, horizon `T=465`, warm-up `W=100` (365-period eval window) — the De Moor / Farrington discounted-return convention.

## Reference instances

`references.rs` carries all 32 Scenario A settings (`m ∈ {2,3,4,5}` × 8 experiments × FIFO/LIFO × waste cost {7,10} × lead time {1,2}). All share `μ=4`, `cv=0.5`, `c=3`, `h=1`, `p=5`, `q̄=10`, horizon 465, warm-up 100. The six the manifest treats as the benchmark roles:

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `de_moor2022_m2_exp1_l1_cp7_lifo` | regime=perishable, issuing=LIFO, m=2, L=1, waste=7, 121 states, role=exact_verification, cv=0.5 | S*=5, VI −1553 | **true (strict_literature_verified)** |
| `de_moor2022_m2_exp2_l1_cp7_fifo` | regime=perishable, issuing=FIFO, m=2, L=1, waste=7, 121 states, role=primary+exact_verification, cv=0.5 | S*=7, VI −1457 | **true (primary; strict_literature_verified)** |
| `de_moor2022_m2_exp4_l1_cp10_fifo` | issuing=FIFO, m=2, L=1, waste=10, 121 states, role=autoresearch_extra | VI −1463 (no Figure-3 policy table) | table_only (re-derivable but no verification.rs assertion) |
| `de_moor2022_m3_exp2_l1_cp7_fifo` | issuing=FIFO, m=3, L=1, waste=7, 1331 states, role=autoresearch_extra | VI −1424 | table_only |
| `de_moor2022_m2_exp6_l2_cp7_fifo` | issuing=FIFO, m=2, **L=2** (first genuine in-transit pipeline), waste=7, 1331 states | VI −1461 | table_only |
| `de_moor2022_m4_exp6_l2_cp7_fifo` | issuing=FIFO, m=4, L=2, ~1.3M states, role=practical_benchmark | VI −1432 / base-stock −1453 (stored anchors) | table_only (exceeds 2000-state exact cap) |

Note: `m=4` and `m=5` experiments 6 and 8 all report `−1432 / −1453` — that duplication is in Farrington Table 3 itself, not a transcription error.

## Baselines

- **Heuristics:** `base_stock` (single order-up-to level S, `q_t = (S − inventory position)^+`); `bsp_low_ew` (low-inventory / estimated-waste base-stock with a threshold). Code in `heuristics/base_stock.rs`, `heuristics/bsp_low_ew.rs`. The base-stock level is tuned on a dedicated heuristic-search seed block; the published best levels are S=7 (FIFO) and S=5 (LIFO).
- **Exact / optimal:** exact tabular value iteration (`value_iteration_mdp.rs`; γ=0.99; midpoint-binned Gamma demand; capped at 2000 states via `bindings.rs`). Re-derives the De Moor optimal-policy table, best base-stock level, and the Farrington Table 3 VI return — only for the four `m=2`, `L=1` (121-state) instances. The `m≥3` / `L=2` instances (1331 … ~1.77M states) exceed the cap and are not re-derived.
- **Published comparators (CONTEXT only):** Farrington 2025 Table 3 VI returns (FIFO −1457±59, LIFO −1553±61) are the **analytic** optimum under midpoint-binned Gamma — a *different estimator* from the Monte-Carlo rollouts, so VI proximity is context, never a "beat." De Moor's own DQN / shaped-DQN comparators are NOT re-implemented here (the repo's learned comparator is the soft tree). Farrington best base-stock FIFO −1474 (reproduced −1475 within tol 1.0).

## Verification

- **Published numbers** (4 instances; m=2/L=1): Farrington 2025 Table 3 VI = FIFO **−1457**, LIFO **−1553**; De Moor best base-stock S=**7** FIFO / S=**5** LIFO; De Moor 9×9 optimal-policy tables (FIFO/LIFO); Farrington best base-stock FIFO **−1474**.
- **Re-run reproduced:** VI **−1457.281** (rounds to −1457) / **−1552.991** (rounds to −1553); S=**7** / S=**5**; `matches_published_policy_table = True`; FlowNet base-stock FIFO observed **−1475** (within tol 1.0). All `matches_published_*` flags `True` this audit. Verdict: **verified_rerun** (genuine VI re-derivation of 3 independent published quantities, not a stored-literal snapshot).
- Re-run via:
  - `ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp2_l1_cp7_fifo')` → `value_iteration_mean_return_rounded`, `best_base_stock_level`, `matches_published_value_iteration_mean_return`, `matches_published_policy_table`, `matches_published_base_stock_level`
  - `ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp1_l1_cp7_lifo')`
  - `ir.perishable_inventory_flownet_policy_verification_summary()` → `summary.all_observed_targets_within_tolerance`
  - In-crate executable assertions: `tests/verification.rs` (reference-set shape, base-stock levels, policy tables, VI returns, observation layout).
- **Verification debts / caveats (state plainly):**
  - **28 of 32 Scenario A rows are TABLE-ONLY** — their Farrington numbers are stored anchors (1331 … ~1.77M states), NOT independently re-derived by the in-repo solver. Do not read them as "verified." Of the 6 representative instances above, 4 (the `m=2/L=1` set) are table_only beyond exact reach; only 2 (`exp1_lifo`, `exp2_fifo`) carry a `verification.rs` assertion. (`exp4_cp10_fifo` is re-derivable at 121 states but has no assertion.)
  - **De Moor "Figure 3" label unconfirmed:** the repo labels the De Moor optimal-policy tables "Figure 3"; the exact published figure number was not independently confirmed (EJOR full text paywalled). The *substance* (the table values, base-stock levels) is verified.
  - **Estimator mismatch:** the VI optimum is the analytic expected discounted return; learned/gate rows are Monte-Carlo means. The optimal base-stock evaluates ~1% worse under Monte-Carlo than under the analytic estimator — a property of the estimator, not the policy. Compare WITHIN one estimator (the gate); treat VI proximity as context.

## Results (learned policy)

- **RESOLVED — seed-robust (≥5 seeds, full budget, 2026-06-06): ROBUST BEATS.** Driving `autoresearch_perishable_inventory.py` over 5 CMA-ES seeds (no code fix needed — the "soft_tree import" brick was stale; scripts already migrated): **FIFO `m2_exp2_l1_cp7` = +1.171% ± 0.002% (5/5; learned reaches the VI optimum)**, **LIFO `m2_exp1_l1_cp7` = +0.840% ± 0.034% (5/5)**, plus extra cells FIFO cp10 +1.43% / LIFO cp10 +0.61% (5/5 each). Margins ≫ cross-seed std → robust beats vs the same-protocol base-stock gate; the single-seed headline below is confirmed seed-robust. (The eval-block-selection sensitivity noted earlier did not flip any FIFO/LIFO headline at N=5.)

Best learned **depth-2 oblique-split, linear-leaf soft tree (21 params)**, warm-started at the encoded best base-stock level (generation-0 reproduces the gate to within its SEM), CMA-ES, selected on a disjoint validation block, scored on 2048 held-out CRN eval seeds (horizon 465, γ=0.99). Versus the same-protocol Monte-Carlo base-stock gate:

| instance | gate (MC) | learned (MC) | Δ vs gate | seed status |
|---|---:|---:|---|---|
| `m2_exp2_l1_cp7_fifo` | −1475.709±0.037 | −1458.509±0.438 | **+1.166%±0.030%**, 5/5 optimizer seeds | **seed-robust** |
| `m2_exp1_l1_cp7_lifo` | −1566.455±0.033 | −1553.552±0.988 | **+0.824%±0.065%**, 5/5 optimizer seeds | **seed-robust** |
| `m2_exp4_l1_cp10_fifo` | −1485.40 | −1464.06 | +21.34 (+1.44%) | **single_seed, NOT yet seed-robust** |
| `m3_exp2_l1_cp7_fifo` | −1435.30 | −1425.20 | +10.09 (+0.70%) | **single_seed, NOT yet seed-robust** |
| `m2_exp6_l2_cp7_fifo` | −1495.44 | −1462.45 | +32.99 (+2.21%) | **single_seed, NOT yet seed-robust** |

- **HONESTY FLAG:** the two exact-anchor `m=2/L=1` rows above are now certified seed-robust over five optimizer seeds in `outputs/perishable_inventory/seed_robust_report.json`. The remaining larger/table-only rows are still from a **single optimizer seed** (`seed=123`; the 2048 seeds are demand-path eval seeds, not optimizer seeds) and must remain at-risk until re-run as mean±std over ≥5 optimizer seeds.
- The exact_slice_report (separate runner) reports `soft_tree_sigmoid_linear` beating the best heuristic FIFO ~15.6 / LIFO ~14 units, with `soft_tree_linear` landing in a worse LIFO basin (an honest negative). Also single-seed.
- The VI-optimum proximity (learned ≈ −1457.90 vs analytic −1457.28 FIFO) is **context only** (mixes estimators).

## Reproduce

```bash
# Verification (exact VI re-derivation of the published quantities):
python3 -c "import invman_rust as ir; s=ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp2_l1_cp7_fifo'); print(s['value_iteration_mean_return_rounded'], s['best_base_stock_level'], s['matches_published_value_iteration_mean_return'], s['matches_published_policy_table'], s['matches_published_base_stock_level'])"
python3 -c "import invman_rust as ir; s=ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp1_l1_cp7_lifo'); print(s['value_iteration_mean_return_rounded'], s['best_base_stock_level'])"
python3 -c "import invman_rust as ir; r=ir.perishable_inventory_flownet_policy_verification_summary(); print(r['summary']['all_observed_targets_within_tolerance'])"

# Learned-policy autoresearch run (single optimizer seed; smoke budget for a wiring check):
python3 /home/nima/code/ml/invman/scripts/perishable_inventory/autoresearch_perishable_inventory.py --reference de_moor2022_m2_exp2_l1_cp7_fifo --budget smoke --seed 123

# Exact-slice benchmark (exact optimum vs tuned base_stock / bsp_low_ew vs CMA-ES soft tree):
python3 /home/nima/code/ml/invman/scripts/perishable_inventory/run_exact_slice_benchmark.py
```

## Pointers & caveats

- **code:** `src/problems/perishable_inventory/env.rs` (`step_state`, `apply_demand_to_inventory` FIFO/LIFO, waste = oldest leftover bucket), `references.rs` (32 Scenario A instances + verification structs), `value_iteration_mdp.rs` (exact tabular VI, 2000-state cap), `heuristics/` (`base_stock`, `bsp_low_ew`), `rollout.rs`, `tests/verification.rs` (executable assertions), `literature/README.md`, `verification/README.md`.
- **scripts:** `scripts/perishable_inventory/` — `autoresearch_perishable_inventory.py` (the working soft-tree CMA-ES runner), `run_exact_slice_benchmark.py` (working), `run_practical_benchmark.py`. NOTE: `run_paper_benchmark.py` and `common.py` are **dead** (import the removed `invman.policies.soft_tree`); use the autoresearch / exact-slice runners.
- **autoresearch:** `policy_search/programs/program_perishable_inventory.md`; ledger `outputs/autoresearch/perishable_inventory_autoresearch/results.tsv`.
- **caveats:**
  - The exact-anchor FIFO/LIFO learned "beats gate" results are seed-robust; the larger/table-only learned rows are still **single optimizer seed** observations and should not be cited as certified beats.
  - The VI optimum and Farrington Table 3 numbers are **analytic** (midpoint-binned Gamma); learned/gate rows are **Monte-Carlo** — only the gate comparison is apples-to-apples.
  - De Moor DQN / shaped-DQN are documented but NOT re-implemented (cross-protocol DRL context).
  - Demand is **Gamma(μ=4, cv=0.5) rounded to integer** (cv = coefficient of variation, not variance).
  - Only the four `m=2/L=1` instances are exact-verified; the rest are table-only anchors.
```
