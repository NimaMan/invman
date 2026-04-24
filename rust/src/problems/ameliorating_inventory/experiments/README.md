# Experiments

This folder is the paper-facing benchmark home for `ameliorating_inventory`.

Planned use:

- define the reported instances for the paper
- compare CMA-ES-optimized learned policies against the carried heuristics
- include an exact or strongest reference comparator when available

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`

Current status:

- no paper-facing experiment suite should be treated as literature-comparable until the executable
  formulation gap to the paper is resolved
