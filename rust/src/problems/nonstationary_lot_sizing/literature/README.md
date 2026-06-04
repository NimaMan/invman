# Literature Overview

## Verification status: literature-verified (published author-repo benchmark
## numbers reproduced by the repo solver+simulator) — with one fidelity caveat

- Dehaybe, Catanzaro & Chevalier (2024), "Deep Reinforcement Learning for
  inventory optimization with non-stationary uncertain demand", European Journal
  of Operational Research 314(2):433-445.
  - DOI: https://doi.org/10.1016/j.ejor.2023.10.007
- HenriDeh/DRL_MMULS single-item branch (the article's public companion code)
  - URL: https://github.com/HenriDeh/DRL_MMULS/tree/single-item
  - Ships the forecast library, the (s,S) testbed code
    (`src/testbed/single-item/sspolicy_test.jl`), and the per-instance benchmark
    CSVs used as the verification anchors.

## What the article models

Single-item, periodic-review stochastic lot-sizing with a rolling demand
forecast: holding cost, shortage/penalty cost, fixed setup cost, procurement
cost, lead time, and either backorders or lost sales. DRL agents take the
forecast window as a policy input and are benchmarked against a rolling
Scarf-style (s,S) dynamic program and a simple (s,S) baseline.

## Verification status (HONEST) — literature_verified = false

The repo rule: a number is literature-verified only when an in-crate test
re-runs the env/solver and reproduces a value PRINTED IN A PAPER within a stated
tolerance. Neither anchor in this family clears that bar:

1. Per-instance benchmark rows (the eight fixed-forecast lost-sales instances).
   These are reproduced from the author's PUBLIC TESTBED CSVs, not from an
   article table:
   - `data/single-item/scarf_testbed_DP_lostsales.csv` (rolling-DP rows, Poisson)
   - `data/single-item/scarf_testbed_simple_lostsales.csv` (simple (s,S), CVNormal)
   produced by `scripts/single-item/experiments/'DP_solve lostsales.jl'` over the
   grid `Iterators.product([2,4,8],[5,10],[10,20,30],[true])`, CV=0.2, H=32. This
   is a reference-implementation match. The article's reported experiment grid
   (`experiment_parameters_lostsales.jl`: leadtimes [8,4,1,0], shortages_ls
   [50,75,100], setups [0,80,1280], CVs [0.1,0.3], horizons [16,8,4]) is a
   DIFFERENT grid, so the carried rows are not the article's tabulated results.

2. Section-4 worked transition (period cost 130 / reward -130). This is carried
   as an INTERNAL `step_state` mechanics / self-consistency check. The EJOR full
   text was not accessible to this repo (paywalled; the OA submitted version on
   the UCLouvain DIAL repository was unreachable), so we do NOT claim -130 is a
   number printed in the article.

## Canonical instance roles

- primary reference instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification problem instance: `constant_10_rolling_dp_reference`

## Benchmark policies carried by the repo

- `simple_s_s`
- `rolling_dp_s_s`
- `lead_time_base_stock`
- `ppo` as literature comparator only

## To upgrade to literature_verified = true later

Obtain the EJOR full text and locate an article-printed per-instance value (a
table cell or a numeric figure annotation), or a confirmed-printed worked-example
number, that this family's env/solver can reproduce; then add an executing
in-crate test asserting reproduction within tolerance and flip the flag with a
precise citation (table/figure + page).
