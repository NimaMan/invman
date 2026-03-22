# invman_rust

Rust extension module for high-throughput inventory-management rollouts.

Current scope:

- native lost-sales rollout kernels
- batched soft-tree population evaluation for CMA-ES
- soft-tree policy support for:
  - `oblique` and `axis_aligned` split types
  - `constant` and `linear` leaf outputs

Build into the project virtualenv with:

```bash
python ../scripts/build_rust_extension.py
```

## Current best native-backed tree result

On the trusted vanilla lost-sales benchmark, the best tree architecture using this native path is:

- oblique split structure
- depth `2`
- linear leaf outputs
- mean cost `4.753725`

This currently outperforms the heuristic baseline `Myopic-2 = 4.8204`.
