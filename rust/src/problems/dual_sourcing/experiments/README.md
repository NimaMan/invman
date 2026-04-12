# Experiments

This folder is the paper-facing benchmark home for `dual_sourcing`.

Current code anchors:

- the Gijs Figure 9 experiment family is defined in `mod.rs`
- benchmark instances and published Figure 9 gap labels live in `../literature/references.rs`
- bounded-DP and heuristic benchmark evaluation live in `bounded_dp.rs`
- the repo-wide batch comparison helper lives in `scripts/dual_sourcing/validate_reference_grid.py`

Python bindings exposed from this folder:

- `invman_rust.dual_sourcing_list_experiment_grids()`
- `invman_rust.dual_sourcing_get_experiment_grid(name)`
- `invman_rust.dual_sourcing_expand_experiment_grid(name)`

When we add checked-in experiment outputs, this folder should carry:

- benchmark notes or manifests for canonical runs
- report snapshots for paper-facing comparisons
- any problem-specific tables derived from the Rust benchmark path
