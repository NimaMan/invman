# Literature Overview

## Verification status: VERIFIED (yes)

The environment faithfully matches the cited model AND the repo reproduces the
published author-repo benchmark numbers on all eight forecast instances. Evidence
is recorded below.

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
  + holding*max(end_inventory, 0) + penalty*unmet_demand. This reproduces the
  paper's Section 4.2 worked transition exactly: with the worked inputs the period
  cost is 130 and the reward is -130 (verification.rs::worked_example_transition_matches_section_4_2).
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
