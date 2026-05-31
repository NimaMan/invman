# Literature Overview

## Verification status: literature-verified (published author-repo benchmark
## numbers reproduced by the repo solver+simulator) — with one fidelity caveat

What is literature-verified vs. self-consistent:

- LITERATURE-VERIFIED (published-number reproduction). The eight reference rows in
  `references.rs` are byte-for-byte the author's public testbed CSVs
  (`HenriDeh/DRL_MMULS`, `single-item` branch, slice leadtime=2/shortage=5/setup=10/
  lostsales/CV=0.2/horizon=32), and the repo's OWN solver + Monte-Carlo simulator
  reproduces every one of those published numbers to within +/-0.17% (table below).
  This is a real re-derivation by a solver, not stored numbers. NOTE: the anchors
  are the author's public CODE-REPO CSVs (the author's `simple` (s,S) and rolling-DP
  (s,S) heuristic baselines), not a numeric table printed in the peer-reviewed EJOR
  article; both author baseline values were independently confirmed against the
  GitHub raw CSVs during the 2026 literature audit (see "Audit trail" below).
- SELF-CONSISTENT-ONLY (one fidelity claim). The "Section 4.2 worked transition"
  test (period cost 130, reward -130) is validated only against the repo's own
  `env.rs::step_state`. The specific worked-example numbers attributed to the paper's
  Section 4.2 could NOT be confirmed against the published article text during the
  2026 audit (the open-access UCLouvain PDF is behind a JS landing page and
  ScienceDirect/ResearchGate returned 403). Treat the worked transition as an
  internal mechanics regression check whose attribution to a printed Section 4.2
  example is unconfirmed by an independent reader.

## Audit trail (2026 literature audit)

- Citation metadata confirmed exact via IDEAS/RePEc
  (https://ideas.repec.org/a/eee/ejores/v314y2024i2p433-445.html): authors Henri
  Dehaybe, Daniele Catanzaro, Philippe Chevalier; EJOR vol. 314, issue 2, pp.
  433-445, 2024; DOI 10.1016/j.ejor.2023.10.007.
- The DRL agent in the paper is PPO (the repo correctly records `ppo` as a
  literature comparator only; CMA-ES soft trees are the repo's own learned policy).
- Author CSV anchors confirmed byte-for-byte against the GitHub raw files:
  - `.../single-item/data/single-item/scarf_testbed_simple_lostsales.csv`
    (forecast_id 1..8 simple_cost = 1252.4885.., 1832.9142.., 2369.6266.., 1824.9849..,
    1869.9016.., 1858.1097.., 1754.7651.., 1964.4607.. — all match `references.rs`)
  - `.../single-item/data/single-item/scarf_testbed_DP_lostsales.csv`
    (forecast_id 1..5 opt_cost = 1215.264, 1711.741, 2072.164, 1675.81, 1680.512 —
    all match `references.rs`)
- Could not verify from source: the Section 4.2 worked-example numbers (see above).

## Primary sources

- Dehaybe, Catanzaro & Chevalier (2024), "Deep Reinforcement Learning for
  inventory optimization with non-stationary uncertain demand", European Journal
  of Operational Research 314(2):433-445.
  - DOI: https://doi.org/10.1016/j.ejor.2023.10.007
  - Single-item Stochastic Lot-Sizing Problem (SISLSP) with fixed ordering cost,
    lead time, rolling forecasts, and both backorder and lost-sales variants.
- HenriDeh/DRL_MMULS, single-item branch (author code + testbed data)
  - URL: https://github.com/HenriDeh/DRL_MMULS/tree/single-item
  - Ships the forecast library, the (s,S) testbed code
    (`src/testbed/single-item/sspolicy_test.jl`), and the per-instance benchmark
    CSVs used as the verification anchors.

## What the repo implements and how it matches the literature

- Order of events per period (env.rs::step_state): place order -> oldest pipeline
  order arrives -> demand realizes -> charge fixed cost K (if an order was placed)
  + holding*max(end_inventory, 0) + penalty*unmet_demand. With the worked inputs the
  period cost is 130 and the reward is -130
  (verification.rs::worked_example_transition_matches_section_4_2). The test name and
  these numbers are attributed to the paper's Section 4.2 worked transition, but that
  attribution was NOT independently confirmed against the published article during the
  2026 audit (PDF inaccessible); the test currently verifies only internal
  self-consistency of `env.rs::step_state`, not a printed paper example.
- simple_s_s heuristic matches the author testbed formula in
  `sspolicy_test.jl` term-for-term:
  - `LTDmean = sum(forecast[1 .. 1+L])`, `LTDstd = sqrt(sum((forecast_i * CV)^2))`
  - `s = quantile(Normal(LTDmean, LTDstd), b/(b+h))`
  - `EOQ = sqrt(2 * mean(forecast) * setup / h)`, `S = s + EOQ`
  - (lead_time_base_stock.rs computes s; simple_ss.rs adds the EOQ to get S.)
- rolling_dp_s_s is the paper's strong dynamic-programming baseline: a finite-horizon
  backward DP on inventory position over the forecast window plus a stationary tail,
  Poisson demand, with the first-period (s,S) levels replayed (rolling_dp.rs).
- Demand models follow the testbed: the "simple" baseline is evaluated under
  CV-Normal demand (CV=0.2); the "DP" baseline is evaluated under Poisson demand.

## Reproduced published numbers (author-repo CSV anchors)

The eight reference rows in `references.rs` are byte-for-byte the author CSVs for
the canonical slice leadtime=2, shortage=5, setup=10, lostsales, CV=0.2, horizon=32:

- simple baseline: `data/single-item/scarf_testbed_simple_lostsales.csv`
- DP baseline:     `data/single-item/scarf_testbed_DP_lostsales.csv`

The repo's own Monte-Carlo simulator reproduces every one of those rows. Measured
at 25,000 replications (seed 1234), the cost differences vs the published values
are (repo - published):

| instance     | simple repro | simple pub | %diff  | dp repro | dp pub  | %diff  |
| ------------ | -----------: | ---------: | -----: | -------: | ------: | -----: |
| constant_5   |      1253.11 |    1252.49 | +0.05% |  1214.92 | 1215.26 | -0.03% |
| constant_10  |      1834.92 |    1832.91 | +0.11% |  1714.15 | 1711.74 | +0.14% |
| constant_15  |      2367.30 |    2369.63 | -0.10% |  2071.94 | 2072.16 | -0.01% |
| seasonal_1   |      1825.71 |    1824.98 | +0.04% |  1678.18 | 1675.81 | +0.14% |
| seasonal_2   |      1870.33 |    1869.90 | +0.02% |  1678.84 | 1680.51 | -0.10% |
| seasonal_4   |      1857.53 |    1858.11 | -0.03% |  1684.87 | 1687.43 | -0.15% |
| growth       |      1753.17 |    1754.77 | -0.09% |  1605.69 | 1603.74 | +0.12% |
| decline      |      1963.09 |    1964.46 | -0.07% |  1837.68 | 1840.87 | -0.17% |

All absolute differences are well within the stored verification tolerance of 35
cost units (and within ±0.17% in relative terms). Reproduced with
`scripts/nonstationary_lot_sizing/run_literature_benchmark.py --replications 25000`.

## Canonical instance roles

- primary reference instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification problem instance: `constant_10_rolling_dp_reference`

## Benchmark policies carried by the repo

- `simple_s_s` (author "simple" baseline, CV-Normal demand)
- `rolling_dp_s_s` (author "DP" baseline, Poisson demand; the strongest comparator)
- `lead_time_base_stock` (repo heuristic comparator)
- `soft_tree` (learned policy; trainable via CMA-ES through the exposed Rust
  rollout binding -- see "Learned-policy comparison" in the parent README)
- `ppo` is recorded only as a literature comparator (the paper's DRL agent)
