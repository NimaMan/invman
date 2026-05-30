# Experiments

This folder is the paper-facing benchmark home for `production_assembly_distribution_network`.

Planned use:

- define reported network topologies and demand settings
- compare learned policies against carried pairwise heuristics
- include small exact comparators where tractable

Current code anchors:

- heuristics in `heuristics/`
- exact small benchmark in `finite_horizon_dp.rs`

Current status:

- no paper-facing experiment suite should be treated as literature-comparable until the executable
  formulation gap to the paper is resolved
