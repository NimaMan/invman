# nonstationary_lot_sizing — benchmark card

**One-line MDP:** State = (rolling demand-forecast window, net inventory, in-transit pipeline); action = order quantity placed this period; one-period cost = fixed setup `K`·1{order>0} + procurement `c`·q + holding `h`·max(end-inv,0) + penalty `b`·unmet; objective = minimize long-run total/mean cost over a `T`-period horizon under lost sales (or backorders).

**Status:** `verified_rerun` — but against the author's **companion-code testbed CSVs** (HenriDeh/DRL_MMULS, `single-item` branch), NOT a peer-reviewed EJOR article table. The in-crate flag `literature_verified = false` is honest and correct.  **Paper:** no dedicated section in `learning_inventory_control_policies_es.tex` (see Pointers & caveats).

## Problem formulation

Single-item, periodic-review stochastic lot-sizing with a *rolling* (nonstationary) demand forecast — Dehaybe, Catanzaro & Chevalier (2024), EJOR 314(2):433–445, DOI 10.1016/j.ejor.2023.10.007. Per-period timing (matches `env.rs::step_state` and the Section 4.2 worked transition):

1. Observe state: forecast window (length `H`, the next `H` mean-demand signals), net inventory, and `L`-slot pipeline of outstanding orders.
2. **Place order** `q ≥ 0` (action).
3. The **oldest pipeline order arrives** (`pipeline[0]`); the new order `q` enters the tail of the pipeline (lead time `L`).
4. **Demand realizes** (CV-Normal or Poisson per baseline, see below). Under lost sales, unmet demand = max(demand − available, 0) and ending inventory is floored at 0; under backorders, ending inventory may go negative and unmet = max(−end-inv, 0).
5. **Charge** one-period cost = `K`·1{q>0} + `c`·q + `h`·max(end-inv,0) + `b`·unmet. Reward = −cost.
6. Forecast window rolls forward by one period (oldest mean drops, `next_forecast_mean` appended).

Long-run objective: minimize mean period cost (the benchmark reports mean cost over `T` periods × Monte-Carlo replications, plus the lost-sales/shortage rate).

## Reference instances

Canonical slice (all eight forecasts): `L=2, b=5, K=10, h=1, c=0, CV=0.2, H=32, T=104`, lost sales, initial net inventory 20. The eight forecast paths are `references.rs::build_forecast_path`.

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `dehaybe2024_lostsales_lt2_b5_k10_constant_5` | regime:lost_sales; forecast:constant_5; L2/b5/K10/h1; cv0.2; H32; T104 | mean demand 5 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_constant_10` **(PRIMARY + verification anchor)** | regime:lost_sales; forecast:constant_10; L2/b5/K10/h1; cv0.2; H32; T104 | mean demand 10 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_constant_15` | regime:lost_sales; forecast:constant_15; L2/b5/K10 | mean demand 15 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_seasonal_1` | regime:lost_sales; forecast:seasonal 104-period; L2/b5/K10 | sinusoid 10±5 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_seasonal_2` | regime:lost_sales; forecast:seasonal 52-period; L2/b5/K10 | sinusoid 10±5 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_seasonal_4` | regime:lost_sales; forecast:seasonal 26-period; L2/b5/K10 | sinusoid 10±5 | false |
| `dehaybe2024_lostsales_lt2_b5_k10_growth` | regime:lost_sales; forecast:linear growth 5→15; L2/b5/K10 | linear ramp up | false |
| `dehaybe2024_lostsales_lt2_b5_k10_decline` | regime:lost_sales; forecast:linear decline 15→5; L2/b5/K10 | linear ramp down | false |
| `constant_10_rolling_dp_reference` **(VerificationProblemInstance)** | verification anchor; regime:lost_sales; forecast:constant_10 | 25,000 reps; tol 35.0 cost / 0.01 shortage | false |
| `WORKED_EXAMPLE_REFERENCE` (Section 4.2, reward −130) | regime:**backorders**; mechanics self-consistency only; L1 | internal `step_state` check | false |
| `retail_like_weekly_trace` (practical) | regime:lost_sales; repo-curated semi-real; L2; H8; T32; demand:poisson | practical dataset | absent (practical dataset, no flag) |

## Baselines

- Heuristics:
  - `simple_s_s` — closed-form (s,S): `s` = quantile of lead-time-demand Normal at `b/(b+h)`, `S = s + EOQ`, `EOQ = sqrt(2·mean(forecast)·K/h)`. Evaluated under **CV-Normal** demand (the author "simple" column). First-period levels for constant_10: `s=33.351246609652, S=47.49338223338295`.
  - `rolling_dp_s_s` — per-period Scarf-style finite-horizon DP over the rolling forecast window, re-solved each period; discount 0.99, 32-period stationary tail; evaluated under **Poisson** demand (the author "DP" column). **STRONGEST baseline.** First-period (s,S) for constant_10: `(28,42)`.
  - `lead_time_base_stock` — repo heuristic, base-stock with no EOQ fixed-cost batching (CV-Normal); reported as an additional comparator.
- Exact / optimal: **none.** There is no exact/global optimum for the rolling-forecast path. `rolling_dp_s_s` is the strongest available comparator and is NOT presented as a global optimum.
- Published comparators (CONTEXT only): the article names a **PPO/DRL** agent (`ppo` is carried as a *name only* in `references.rs`). **No PPO/DRL cost number is carried** — the EJOR full text was inaccessible (paywalled; OA DIAL copy unreachable). Treat any PPO mention as cross-protocol context, never a head-to-head beat.

## Verification

- Published number (author **companion-code testbed CSV**, `scarf_testbed_simple_lostsales.csv` / `scarf_testbed_DP_lostsales.csv`), constant_10:
  - simple (s,S): mean_cost = **1832.9142436489014**, shortage = 0.0029443487165113735 (CV-Normal)
  - rolling-DP: mean_cost = **1711.741**, shortage = 0.04793465748308879 (Poisson)
- **Re-run reproduced** (RAYON_NUM_THREADS=4, 25,000 reps):
  - simple_s_s constant_10: **1834.918166 (+0.109%)**, shortage 0.002871 — within 35.0 cost / 0.01 shortage tolerance.
  - rolling_dp constant_10: **1714.147560 (+0.141%)**, shortage 0.048469.
  - simple_s_s levels EXACT `s=33.351246609652, S=47.493382233383`; rolling_dp first-period `(28,42)` EXACT.
  - growth simple_s_s: 1753.169 (−0.091%). Worked transition reward −130 confirmed (internal mechanics).
  - **Verdict: verified_rerun vs companion CSV** (repo flag `literature_verified=false` is honest).
- DEBT / caveats:
  - This is a **reference-implementation match, NOT a peer-reviewed article table.** The testbed grid `product([2,4,8],[5,10],[10,20,30],[true])`, CV=0.2, H=32 also differs from the article's reported experiment grid (`leadtimes [8,4,1,0], shortages_ls [50,75,100], setups [0,80,1280], CVs [0.1,0.3], horizons [16,8,4]`).
  - The Section 4.2 worked transition (period cost 130 / reward −130) is a **self-consistency-only** check of `env.rs::step_state` (uses backorders, L=1, h=1, b=10, K=100). It is NOT confirmed to be a number printed in the article.
  - The **non-constant rolling-DP cases (growth/decline/seasonal) did NOT finish in ~2 min** during the re-run audit (per-period 32-period DP × 104 periods); only constant-forecast DP rows were re-run within budget.

## Results (learned policy)

- **CARRIED, but SINGLE-SEED — NOT yet seed-robust (`at_risk`):** A learned CMA-ES soft tree is reported to beat `rolling_dp_s_s` on all 8/8 forecasts by −6.5% to −15.5% (e.g. constant_5 1026.3 vs 1214.9 = −15.52%; constant_10 1539.0 vs 1714.1 = −10.22%), and to be the cheapest policy on 5/8 instances (on seasonal_2/growth/decline it beats DP but trails `lead_time_base_stock` by +0.35% to +3.46%). **`seed_reporting = single_seed`.** Per the repo's seed-robust reporting standard (mean±std over ≥5 optimizer seeds, never single/best-of-N), this beat is **NOT yet seed-robust** and must not be carried as an established result.
- **Seed-robust / verified:** the heuristic baselines reproduce the author-CSV rows within ~0.11–0.14% (≤0.17% bound). This is the `multi_seed_mean_std`, not-at-risk result.

## Reproduce

```bash
# Heuristic reproduction (constant cases fast; non-constant rolling-DP is slow)
RAYON_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_literature_benchmark.py --replications 25000

# Closed-form / first-period levels for constant_10 (simple (s,S) and rolling-DP)
RAYON_NUM_THREADS=4 python -c "import invman_rust as ir; f=[10.0]*136; print(ir.nonstationary_lot_sizing_simple_s_s_levels(f[:32],2,1.0,5.0,10.0,'cv_normal',0.2)); print(ir.nonstationary_lot_sizing_rolling_dp_s_s_levels(f[:32],2,1.0,5.0,10.0,'poisson',0.99,32))"

# Learned soft-tree (single-seed; NOT seed-robust as run here)
RAYON_NUM_THREADS=2 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_literature_benchmark.py --learned --tree_depth 2 --leaf_type linear --action_cap 100 --generations 150 --popsize 48 --learned_replications 10000 --output_json /tmp/learned.json

# Practical curated-trace benchmark
RAYON_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_practical_benchmark.py

# In-crate verification tests
cargo test -p invman_rust nonstationary_lot_sizing
```

## Pointers & caveats

- code: `src/problems/nonstationary_lot_sizing/` — `env.rs` (MDP/`step_state`), `references.rs` (8 forecast instances + worked transition + verification instance), `heuristics/` (`simple_ss.rs`, `rolling_dp.rs`, `lead_time_base_stock.rs`), `demand.rs` (CvNormal / Poisson), `rollout.rs`, `bindings.rs`; tests in `tests/verification.rs`; literature evidence in `literature/README.md`; verifier contract in `verification/README.md`; practical dataset in `practical/datasets/retail_like_weekly_trace.json`.
- scripts: `scripts/nonstationary_lot_sizing/run_literature_benchmark.py`, `scripts/nonstationary_lot_sizing/run_practical_benchmark.py`.
- autoresearch: **no `autoresearch/program_nonstationary_lot_sizing.md` exists** (the autoresearch program for this family has not been written; design recipe lives in `autoresearch/POLICY_DESIGN_GUIDELINES.md`).
- **Paper caveat (important):** `learning_inventory_control_policies_es.tex` has **no dedicated section** for this system. Its two "nonstationary" mentions (≈ lines 2477, 4215) refer to the multi-echelon *divergent* instance's resampled mean `α∼Uniform[5,15]`, not to this lot-sizing family. The Dehaybe (2024) citation (`dehaybe2024deep`) appears only in a *different* paper file, `paper/invman_lostsales.tex`, as a literature-survey reference. So the "§ of the ES paper" pointer in the card template does not resolve to a real section here.
- **Demand-convention caveat:** the two strongest baselines are evaluated under *different* demand models by design — `simple_s_s` under CV-Normal (cv=0.2), `rolling_dp_s_s` under Poisson — matching the author testbed's separate "simple" and "DP" CSVs. Cost gaps between them mix policy and demand-model differences.
- **Verification debt to upgrade:** obtain the EJOR full text, locate an article-printed per-instance value (table cell / figure annotation) reproducible by this env/solver, add an executing in-crate assertion within tolerance, then flip `literature_verified` to true with a precise table/figure + page citation.
- The existing `README.md` in this folder is self-consistent with the code (it states `literature_verified = false` and explains the companion-CSV vs article-table distinction); however its top headers say "Verification status: literature-verified" before later correcting to "literature_verified = false (HONEST)" — a mildly contradictory framing that this card resolves in favor of `verified_rerun-against-companion-CSV / literature_verified=false`.
