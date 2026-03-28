# Fixed-Order-Cost Literature Subset Baseline

This file summarizes the first repo-native benchmark baseline for the fixed-order-cost lost-sales extension.

## Source

- grid name: `literature_subset_poisson_mu5`
- raw results: `fixed_order_cost_literature_subset_poisson_mu5.json`
- generator: `./.venv/bin/python invman/scripts/benchmark_fixed_order_cost_grid.py`

## Headline Results

- number of instances: `16`
- modified `(s, S, q)` not worse than `(s, S)` on `13 / 16` instances
- modified `(s, S, q)` not worse than `(s, nQ)` on `12 / 16` instances
- mean relative improvement versus `(s, S)`: `0.6692%`
- mean relative improvement versus `(s, nQ)`: `0.1973%`

## Selected Instances

- `lit_pois_mu5_l4_p4_k5`
  - `(s, S)`: `(21, 28)` with mean cost `9.3692`
  - `(s, nQ)`: `(21, 8)` with mean cost `9.212392`
  - modified `(s, S, q)`: `(21, 29, 8)` with mean cost `9.202917`
- `lit_pois_mu5_l4_p19_k25`
  - `(s, S)`: `(26, 42)` with mean cost `21.0202`
  - `(s, nQ)`: `(26, 18)` with mean cost `21.000342`
  - modified `(s, S, q)`: `(26, 41, 19)` with mean cost `20.978817`
- `lit_pois_mu5_l1_p4_k5`
  - `(s, S)`: `(8, 14)` with mean cost `8.268117`
  - `(s, nQ)`: `(8, 7)` with mean cost `8.304767`
  - modified `(s, S, q)`: `(8, 14, 11)` with mean cost `8.230108`
- `lit_pois_mu5_l1_p19_k25`
  - `(s, S)`: `(10, 26)` with mean cost `18.641208`
  - `(s, nQ)`: `(10, 18)` with mean cost `18.65345`
  - modified `(s, S, q)`: `(10, 27, 21)` with mean cost `18.654008`

## Interpretation

These are repo-computed benchmark values, not published literature targets. The literature target for this problem family is the qualitative ranking and average-gap pattern reported by Bijvank, Bhulai, and Huh (2015), while this file provides the reproducible per-instance baseline we can use for regression testing and policy comparisons inside this repo.
