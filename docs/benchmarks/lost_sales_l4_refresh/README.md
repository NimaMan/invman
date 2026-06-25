# Lost-Sales L=4 Refresh

Canonical benchmark instance:

- problem: vanilla lost sales
- lead time `L=4`
- shortage cost `p=4`
- holding cost `h=1`
- demand `~ Poisson(5)`
- `max_order_size=20`

## Trusted heuristic baseline

The reference instance remains stable after the Rust crate refactor. Running
`scripts/validate_reference_instance.py` on `vanilla_l4_p4_poisson5` still reproduces the expected
heuristic numbers at horizon `100000` with `3` seeds and `track_demand=True`:

- Myopic-1: `5.0641`
- Myopic-2: `4.8204`
- SVBS: `5.8349`
- optimal reference: `4.73`

These are the core trust anchors for the vanilla problem.

## Fresh post-refactor reruns

All fresh reruns below use the same benchmark instance and evaluate over horizon `100000` with `3`
seeds.

| Policy | Backend | Train budget | Mean cost | Interpretation |
| --- | --- | ---: | ---: | --- |
| Soft tree, oblique depth-2, linear leaves | Rust | 2000 ES iterations | `4.7658` | Healthy rerun; still clearly better than Myopic-2 |
| Linear categorical quantity | Python | 2000 ES iterations | `5.0049` | Works, but weaker than the old locked linear result |
| NN `8x8` categorical quantity | Python | 1000 ES iterations | `5.2504` | Smoke rerun only; not competitive on this budget |

Fresh result files:

- `invman/outputs/results/refresh_soft_tree_rust_l4_p4_pois5_2k_pop10.json`
- `invman/outputs/results/refresh_linear_l4_p4_pois5_2k_pop10.json`
- `invman/outputs/results/refresh_nn8x8_l4_p4_pois5_1k_pop10.json`

## Locked historical references

The fresh reruns confirm the environment and the Rust soft-tree path are sound, but the linear and NN
backbones are still more sensitive to seed/budget than the tree path. For that reason, these remain the
stronger historical references for the canonical vanilla benchmark:

- historical linear reference: `4.8066`
- historical best soft tree: `4.7537`
- paper Linear-8: `4.777`
- paper Neural network-20: `4.758`
- paper Neural network-8: `4.752`

The paper values come from `invman_paper (revision)/invman.tex` for the `L=4`, `Poisson`, shortage-cost
`4` setting.

## Current interpretation

- The vanilla lost-sales environment is still correct.
- The Rust-backed soft-tree rollout path is still correct and lands in the expected range.
- The current Python linear and NN reruns underperform their locked historical references, so they should
  not yet replace those older benchmark anchors.
- For future policy work, the best stable vanilla learned-policy reference is still the oblique depth-2
  soft tree with linear leaves.
