# Dual sourcing — benchmark card

**One-line MDP:** state = reduced post-decision vector `(I_t, q^r_{t-l_r+1}, ..., q^r_{t-1})` (net inventory after the regular arrival, plus the in-transit regular pipeline); action = integer order pair `(q^r_t, q^e_t)` within caps `(\bar q_r, \bar q_e) = (12, 12)`; one-period cost `c_t = c_r q^r_t + c_e q^e_t + h (J_t)^+ + b (J_t)^-` with end-of-period net inventory `J_t = I_t + q^e_t - D_t`; objective = minimize the long-run average expected cost.

**Status:** verified_rerun (peer-reviewed published number — Gijsbrechts et al. 2022 Figure 9 optimality gaps), reproduced by re-run on the two `l_r=2` rows; the `l_r=3,4` rows carry a standing verification debt (ledger D4 — prior-date validated 2026-06-04, NOT re-run this audit). **Paper:** §"Dual sourcing" (`\label{sec:dual-sourcing}`, formulation/policy/results §§ around lines 1119–1389) of `paper/learning_inventory_control_policies_es.tex`.

## Problem formulation

Single-item, periodic-review dual-sourcing model of Gijsbrechts et al. (2022). Two suppliers replenish one item: a slow **regular** supplier (lead time `l_r >= 1`, unit cost `c_r`) and a fast **expedited** supplier (lead time `l_e = 0`, unit cost `c_e > c_r`). Demand is i.i.d. `D_t ~ U{0,1,2,3,4}`. Unmet demand is **backordered** (not lost), so net inventory may go negative.

- **Timing of a period `t`:** the regular order placed `l_r` periods earlier, `q^r_{t-l_r}`, arrives and merges into net inventory, giving start-of-period net inventory `I_t`. The controller then places `q^r_t` (arrives `l_r` periods later) and `q^e_t` (arrives immediately, since `l_e = 0`). Demand `D_t` is served from on-hand `I_t + q^e_t`. Holding and backorder costs are charged on end-of-period net inventory.
- **State:** because only the expedited lead time is zero, only the regular pipeline must be tracked. The MDP state is the `l_r`-dimensional reduced post-decision vector `S_t = (I_t, q^r_{t-l_r+1}, ..., q^r_{t-1})`. For `l_r=2` this is `(I_t, q^r_{t-1})`. (Implemented in `env.rs::step_state`, with `reduced_state[0]` the net inventory.)
- **Action:** `a_t = (q^r_t, q^e_t)`, `q^r_t in {0..\bar q_r}`, `q^e_t in {0..\bar q_e}`, caps `\bar q_r = \bar q_e = 12`.
- **Transition:** `J_t = I_t + q^e_t - D_t`; `I_{t+1} = J_t + q^r_{t-l_r+1}`; the pipeline shifts forward and the new regular order `q^r_t` enters the last slot. For `l_r=1` the state collapses to `I_{t+1} = I_t + q^e_t - D_t + q^r_t`.
- **One-period cost:** `c_t = c_r q^r_t + c_e q^e_t + h (J_t)^+ + b (J_t)^-` (env.rs `epoch_cost`, with `c_r = 100`, `h = 5`, `b = 495`).
- **Objective:** minimize `\bar C(theta) = lim_{T->inf} (1/T) sum_t E[c_t]`, the long-run average expected cost, estimated by simulation with a warm-up period.

## Reference instances

The six instances are the full cross of regular lead time `l_r in {2,3,4}` and expedited unit cost `c_e in {105,110}`. All share `l_e=0`, `c_r=100`, `h=5`, `b=495`, demand `U{0,1,2,3,4}`, and order caps `12`. (`literature/references.rs::DUAL_SOURCING_REFERENCE_INSTANCES`.)

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `dual_l2_ce105` (primary unit-test instance) | regime:dual_sourcing_backlog; leadtime:`l_r`=2; `c_e`=105; demand:U0-4; `c_r`=100; `h`=5; `b`=495 | l_r=2, l_e=0, c_r=100, c_e=105, h=5, b=495, caps 12 | true (grid hardcoded for all 6 in `experiments/mod.rs`; `references.rs` has NO per-instance flag field) |
| `dual_l2_ce110` | leadtime:`l_r`=2; `c_e`=110; demand:U0-4 | l_r=2, c_e=110, else as above | true |
| `dual_l3_ce105` | leadtime:`l_r`=3; `c_e`=105 | l_r=3, c_e=105, else as above | true |
| `dual_l3_ce110` | leadtime:`l_r`=3; `c_e`=110 | l_r=3, c_e=110, else as above | true |
| `dual_l4_ce105` | leadtime:`l_r`=4; `c_e`=105 | l_r=4, c_e=105, else as above | true |
| `dual_l4_ce110` (primary reference instance) | leadtime:`l_r`=4; `c_e`=110; demand:U0-4 | l_r=4, c_e=110, else as above | true |

## Baselines

- **Heuristics** (`heuristics.rs`): `single_index`, `dual_index`, `capped_dual_index` (strongest; used as the optimal proxy), `tailored_base_surge`. Each is found by **exhaustive grid search over its target parameters on a fixed demand path** with a warm-up ratio: single/dual-index sweep `(s_e, s_r)` with `s_e <= s_r` up to a target upper bound; capped-dual-index additionally sweeps the regular cap `cap_r in 0..=12`; tailored-base-surge sweeps `(surge_level, regular_qty)`. (`search_two_param_policy`, `search_capped_dual_index_from_demands`, `search_tailored_base_surge_from_demands`.)
- **Exact / optimal:** bounded DP `solve_bounded_average_cost_optimal_policy` (relative value iteration over the truncated inventory box `[-12, 24]`, `bounded_dp.rs`). This is **NOT a proof-level unbounded optimum**: for `l_r=4` the truncated value sits ~0.2% *below* the heuristics, making it unusable as a denominator there. Therefore `capped_dual_index` is used as the optimal proxy (CDI's published gap to the bounded-DP optimum is `<=0.11%` on every row).
- **Published comparators (CONTEXT only; cross-protocol):** Gijsbrechts et al. (2022) Figure 9 optimality-gap labels (% above the bounded-DP optimum). There is **NO published absolute-cost table** — only the Figure 9 bar labels. Per-row gaps:
  - `dual_l2_ce105`: CDI 0.00 / DI 0.11 / SI 0.56 / TBS 0.06 / **A3C 0.52**
  - `dual_l2_ce110`: CDI 0.03 / DI 0.18 / SI 1.03 / TBS 0.99 / **A3C 0.80**
  - `dual_l3_ce105`: CDI 0.00 / DI 0.27 / SI 0.98 / TBS 0.01 / **A3C 0.82**
  - `dual_l3_ce110`: CDI 0.06 / DI 0.36 / SI 2.11 / TBS 0.71 / **A3C 0.51**
  - `dual_l4_ce105`: CDI 0.00 / DI 0.36 / SI 1.43 / TBS 0.00 / **A3C 1.85**
  - `dual_l4_ce110`: CDI 0.11 / DI 0.49 / SI 2.44 / TBS 0.58 / **A3C 1.33**
  - **A3C is a deep-RL comparator (cross-protocol); it is CONTEXT, never something this card claims to "beat."** Reaching CDI's `<=0.11%` band is below A3C's `0.51-1.85%` range, but that is a published-context comparison, not a like-for-like protocol match.

## Verification

- **Published number:** Gijsbrechts Figure 9 optimality gaps. For `dual_l2_ce105`: CDI 0.00%, DI 0.11%, SI 0.56%, TBS 0.06%. For `dual_l2_ce110`: CDI 0.03%, DI 0.18%, SI 1.03%, TBS 0.99%.
- **Re-run reproduced (this audit, 10.4s):** `dual_l2_ce105` optimal_avg = **216.770**; CDI gap **0.0058%** (pub 0.00), TBS **0.0615%** (pub 0.06), DI **0.1164%** (pub 0.11), SI **0.5675%** (pub 0.56). `dual_l2_ce110` optimal_avg = **219.733**; CDI **0.0301%** (pub 0.03), DI **0.1784%** (pub 0.18), TBS **0.9874%** (pub 0.99), SI **1.0316%** (pub 1.03). **All within <=0.0075 percentage points.** Verdict: **verified_rerun** (bounded-DP gaps reproduced; no published *absolute* cost exists to target).
  - Re-run via: `invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce105', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2)` (and the same for `dual_l2_ce110`).
- **DEBT (ledger D4):** the `l_r=3,4` rows are `#[ignore]`d (minutes-scale) and were **NOT re-run this audit**; they were externally validated on 2026-06-04 via the batch script but should be treated as faithful + externally-validated, not freshly re-run, until a dated re-run artifact is wired.
- **DEBT (ledger D5, latent):** the `figure_9_gap_labels_are_frozen` / drift-guard tests in `verification/tests.rs` assert carried `==` published literals and do **NOT** execute the env. A frozen snapshot is NOT verification — the canonical verification is the executing `l_r=2` test; the drift guard is a drift guard only.

## Results (learned policy)

- **Headline (seed-robust, multi-seed mean/std):** the learned soft tree **MATCHES** the capped-dual-index optimal proxy on **all 6 instances**, landing within CDI's own published `<=0.11%` band. This is the robust match floor (manifest `seed_reporting: multi_seed_mean_std`, `at_risk: false`). Reported as a *match*, not an improvement. Reaching CDI's near-optimal band sits below the published A3C gaps (`0.51-1.85%`), but that A3C comparison is cross-protocol context. Best-policy CRN table (70 shared seeds, horizon `6e4`) — learned vs CDI proxy:
  - `l_r=2,c_e=105`: CDI 216.806 / learned 216.806 (+0.000, match)
  - `l_r=2,c_e=110`: CDI 219.821 / learned 219.800 (-0.009, match)
  - `l_r=3,c_e=105`: CDI 216.900 / learned 216.906 (+0.003, match)
  - `l_r=3,c_e=110`: CDI 220.483 / learned 220.483 (+0.000, match)
  - `l_r=4,c_e=105`: CDI 216.914 / learned 216.918 (+0.002, match)
  - `l_r=4,c_e=110`: CDI 220.879 / learned 220.789 (-0.041, match)
- **"Beats CDI on 2 of 6 rows" (`dual_l2_ce110` by -0.009%, `dual_l4_ce110` by -0.041%): single-seed, NOT yet seed-robust.** The manifest marks this `seed_reporting: single_seed`, `at_risk: true` (held-out re-verified on disjoint seeds, but not a mean±std over >=5 optimizer seeds). The margins are economically negligible and sit inside CDI's own `<=0.11%` optimality band; the paper deliberately frames them as *matches*, not wins.
- **Factor-screen negative gaps vs best heuristic** (`dual_l4_ce105` -0.1052%, `dual_l2_ce105` axis-linear -0.0621%): **single-seed, NOT yet seed-robust** (manifest `single_seed`, `at_risk: true`). Treat as exploratory, not as the learned policy genuinely beating the heuristic.

## Reproduce

```bash
# l_r=2 reproduce of Figure-9 gaps (fast, ~10s each)
python -c "import invman_rust; r=invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce105', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2); print(r['optimal']['average_cost']); [print(h['policy_name'], h['optimality_gap_pct'], h['published_optimality_gap_pct']) for h in r['heuristics']]"
python -c "import invman_rust; print(invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce110', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2))"

# validate the full reference grid against published Figure-9 gaps
python /home/nima/code/ml/invman/scripts/dual_sourcing/validate_reference_grid.py
python /home/nima/code/ml/invman/scripts/dual_sourcing/validate_reference_grid.py --references dual_l2_ce105 dual_l2_ce110

# l_r=3,4 rows (minutes-scale, #[ignore]d — the D4 debt)
cargo test -p invman_rust dual_sourcing -- --ignored

# learned-policy factor screen across all six rows
python /home/nima/code/ml/invman/autoresearch/dual_sourcing_policy_search/run_factor_screen.py
```

## Pointers & caveats

- **Code:** `src/problems/dual_sourcing/env.rs` (MDP: `step_state`, `epoch_cost`), `heuristics.rs` (4 structured heuristics + grid search), `bounded_dp.rs` (truncated relative value-iteration reference), `policies.rs` / `rollout.rs` (capped-dual-index decoder + rollout), `bindings.rs` (`dual_sourcing_reference_benchmark_summary`), `literature/references.rs` (6 instances + Figure-9 gap labels), `verification/tests.rs` (executing `l_r=2` checks + frozen drift guards).
- **Scripts:** `scripts/dual_sourcing/` (`validate_reference_grid.py`, `benchmark_full_suite.py`, `dual_sourcing_benchmark_lib.py`, `autoresearch_dual_sourcing*.py`, `sweep_policy_variants.py`).
- **Autoresearch:** `autoresearch/program_dual_sourcing.md`; canonical search surface `autoresearch/dual_sourcing_policy_search/` (`run_factor_screen.py`, `factor_screen_results.md`, `summarize_factor_screen.py`, `README.md`).
- **Caveat — A3C is cross-protocol context:** Gijsbrechts' A3C gaps (0.51-1.85%) are a deep-RL comparator. Matching CDI lands below them, but A3C is never a like-for-like "beats" claim here.
- **Caveat — bounded DP is not a proof-level optimum:** it is a truncated finite-state reference; for `l_r=4` it sits ~0.2% below the heuristics and is unusable as a denominator. CDI is the optimal proxy throughout, justified by its `<=0.11%` published gap.
- **Caveat — "beats CDI" / factor-screen negatives are single-seed:** the only seed-robust result is the *match* on all 6 rows. The two -0.009%/-0.041% "beats" and the factor-screen negatives are single-seed (`at_risk`), economically negligible, and inside CDI's own optimality band; do not promote them to wins.
- **Caveat — action geometry / warm-start:** the learned policy lives in factorized capped-dual-index coordinates `(s_e, Delta_r, \bar c_r)` with `s_r = s_e + Delta_r`; a raw direct-order decoder cannot express the CDI control. CMA-ES is warm-started at the encoded CDI solution (the bottleneck was optimization, not representation).
- **Caveat — README vs card:** the pre-existing `README.md` describes the same family but predates the audit's honest framing; it does not contradict the code, but the verification status and seed-robustness caveats in this card (D4/D5 debts, single-seed "beats") supersede any softer reading of the README. The README is left untouched.
