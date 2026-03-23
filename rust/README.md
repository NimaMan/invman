# invman_rust

Rust extension module for high-throughput inventory-management rollouts.

Current scope:

- native rollout kernels for:
  - lost sales
  - fixed-order-cost heuristic search
  - dual sourcing
  - multi-echelon
- batched soft-tree population evaluation for CMA-ES
- soft-tree policy support for:
  - `oblique` and `axis_aligned` split types
  - `constant` and `linear` leaf outputs

## Source layout

The crate mirrors the Python package structure:

- `src/core/`
  - shared Rust-native policy/runtime pieces
  - currently the generic soft-tree implementation
- `src/problems/<problem>/`
  - problem-local environment transitions
  - rollout kernels
  - heuristic search
  - problem-specific action mappings when needed

Current problem modules:

- `src/problems/lost_sales/`
- `src/problems/lost_sales_fixed_order_cost/`
- `src/problems/dual_sourcing/`
- `src/problems/multi_echelon/`

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
