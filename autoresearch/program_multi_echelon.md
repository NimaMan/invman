# multi-echelon autoresearch

This is the multi-echelon counterpart to the lost-sales and fixed-cost autoresearch programs.

## Benchmark

Primary screening instance:

- `multi_echelon_setting2`
- one warehouse and `K = 10` retailers
- `l_w = 5`
- `l_r = 3`
- rounded normal retailer demand with `mu = 0`, `sigma = 20`
- same-day expedite probability `P_w = 0.8`

The benchmark heuristic is fixed:

- constant base-stock policy on the literature action grid

## Intended search surface

- `invman/policies/`
- `rust/src/policies/`
- `rust/src/rollout/`
- limited support code needed to wire discrete-grid vector actions into training

## Budgets

Use the budgets from `scripts/autoresearch_multi_echelon.py`:

- `screening`
- `full`

## Goal

Lower the learned-policy cost on the primary multi-echelon instance while preserving a clean
general policy pipeline.

Current smoke baseline:

- learned tree: `3776.45`
- best constant base-stock benchmark: `3776.45`
