# Lost Sales

This package is the canonical home for the vanilla lost-sales problem.

Core files:

- `env.py`: simulator and rollout helpers
- `heuristics.py`: Myopic-1, Myopic-2, SVBS, and supporting utilities
- `reference_instances.py`: trusted benchmark instance definitions
- `problem_info.py`: literature-style reference values used by the repo
- `benchmark.py`: problem-level heuristic benchmarking helpers
- `experiment_spec.py`: paper-style learned-policy suite definitions for the vanilla problem

## Canonical benchmark

The trusted regression target in this repo is:

- `L=4`
- shortage cost `p=4`
- holding cost `h=1`
- demand `~ Poisson(5)`

Reference numbers:

- optimal: about `4.73`
- Myopic-2: about `4.82`
- Myopic-1: about `5.06`
- SVBS: about `5.83`

## Learned-policy status

Locked historical references on the canonical instance:

- linear policy: `4.8066`
- best soft tree: `4.7537`
- paper Linear-8: `4.777`
- paper Neural network-20: `4.758`
- paper Neural network-8: `4.752`

Fresh post-refactor reruns:

- Rust-backed soft tree, oblique depth-2 with linear leaves: `4.7658`
- linear categorical quantity policy: `5.0049`
- NN `8x8` categorical quantity smoke rerun: `5.2504`

Interpretation:

- the environment and heuristic layer remain trusted
- the Rust-backed soft-tree path remains trusted
- the linear and NN backbones still need retuning or native porting before their fresh reruns should
  replace the older benchmark anchors

For the detailed refresh note, see `../../../docs/benchmarks/lost_sales_l4_refresh.md`.

## Experiment surface

Stable vanilla lost-sales benchmark entrypoints now come in two modes:

- `scripts/lost_sales/benchmark_canonical_suite.py`: one canonical preflight instance
- `scripts/lost_sales/benchmark_full_suite.py`: full 20-instance literature-aligned grid

The full grid follows the instance family from Xin (2020):

- lead times `L in {2,4,6,8,10}`
- shortage costs `p in {4,19}`
- demand distributions `{Poisson, Geometric}`
- mean demand `5`
