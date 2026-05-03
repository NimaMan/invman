# Experiments

This folder is the paper-facing benchmark home for `procurement_removal_inventory`.

Planned use:

- define reported procurement/removal instances
- compare CMA-ES-optimized learned policies against carried interval-stock heuristics
- include reduced exact comparators where tractable

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
