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
  formulation gap to the paper is resolved (see `../literature/README.md` for the term-by-term gap)

Feasible-now benchmark (repo-native, NOT literature-comparable):

- `scripts/ameliorating_inventory/benchmark_repo_native_instance.py` runs the carried heuristics
  on `PRIMARY_REFERENCE_INSTANCE` via the installed bindings and prints discounted cost/profit.
- It records the two blockers to a full optimal-vs-heuristic-vs-learned comparison:
  - exact DP (`finite_horizon_dp.rs`) is `#[cfg(test)]` and has no Python binding
  - no checked-in trained soft-tree parameters (learned row needs CMA-ES training)
