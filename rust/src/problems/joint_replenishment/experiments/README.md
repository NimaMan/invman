# Experiments

This folder is the paper-facing benchmark home for `joint_replenishment`.

Planned use:

- define reported multi-item instances
- compare CMA-ES-optimized learned policies against the carried heuristics
- include the reduced exact DP comparator when tractable

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
