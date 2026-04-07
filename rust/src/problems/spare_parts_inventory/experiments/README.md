# Experiments

This folder is the paper-facing benchmark home for `spare_parts_inventory`.

Planned use:

- define reported spare-parts instances
- compare CMA-ES-optimized learned policies against carried heuristics
- include reduced exact comparators where tractable
- keep exact literature-validation benchmarks separate from the learned-policy repairable-control
  experiments when the formulation is not the same MDP family

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
