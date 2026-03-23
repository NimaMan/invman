# dual-sourcing autoresearch

This is the dual-sourcing counterpart to the lost-sales and fixed-cost autoresearch programs.

## Benchmark

Primary screening instance:

- `dual_l4_ce110`
- regular lead time `l_r = 4`
- expedited lead time `l_e = 0`
- demand uniform on `{0,1,2,3,4}`
- `h = 5`
- `b = 495`
- `c_r = 100`
- `c_e = 110`

The benchmark heuristics are fixed:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

## Intended search surface

- `invman/policies/`
- `rust/src/policies/`
- `rust/src/rollout/`
- limited support code needed to wire vector-action trees into training

## Budgets

Use the budgets from `scripts/autoresearch_dual_sourcing.py`:

- `screening`
- `full`

## Goal

Lower the learned-policy cost on the primary dual-sourcing instance while preserving a clean
general policy pipeline.

Current smoke baseline:

- learned tree: `249.84`
- best heuristic baseline: capped dual-index `220.73`
