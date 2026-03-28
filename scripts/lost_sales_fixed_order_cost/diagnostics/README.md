# Diagnostics Scripts

This folder holds ad hoc inspection and sweep helpers that are useful during policy-design work but
are not part of the stable experiment surface.

These scripts are intentionally kept out of the top-level `scripts/` directory because they are not:

- part of the curated `numerical_experiments` catalog
- part of the benchmark regeneration path
- part of the paper table export path

Current contents:

- `analyze_policy.py`: inspect the action pattern of a trained fixed-cost policy
- `sweep_nn.py`: run a small diagnostic NN sweep for fixed-cost lost sales
