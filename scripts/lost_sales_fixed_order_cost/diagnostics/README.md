# Diagnostics Scripts

This folder holds ad hoc inspection and sweep helpers that are useful during policy-design work but
are not part of the stable experiment surface.

These scripts are intentionally kept out of the top-level fixed-cost script directory because they are not:

- part of the curated `numerical_experiments` catalog
- part of the stable benchmark regeneration path
- part of the manuscript update path

Current contents:

- `analyze_policy.py`: inspect the action pattern of a trained fixed-cost policy
- `sweep_nn.py`: run a small diagnostic NN sweep for fixed-cost lost sales
- `benchmark_heuristics_grid.py`: run heuristic-only grid sweeps without learned policies
- `compare_search_backends.py`: compare Python and Rust heuristic-search backends on one instance
