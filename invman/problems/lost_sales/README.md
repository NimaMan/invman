# Lost Sales

This package is the canonical home for the vanilla lost-sales problem.

## Literature guidance

### Primary references

- Joren Gijsbrechts, Robert N. Boute, Jan A. Van Mieghem, and Dennis J. Zhang,
  *Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual
  Sourcing, and Multi-Echelon Problems*, Manufacturing & Service Operations Management, 2022.
- DOI: <https://doi.org/10.1287/msom.2021.1064>
- Companion code: <https://github.com/JorenGijsbrechts/DRL_A3C_inventory>

The broader repo benchmark grid also follows the single-item lost-sales family used in Xin (2020),
which is what the full vanilla benchmark surface here is aligned to.

### Published problem family

The Gijsbrechts DRL paper uses six Zipkin-style lost-sales settings:

- lead time `L in {2, 3, 4}`
- shortage penalty `p in {4, 9}`
- holding cost `h = 1`
- ordering cost `c = 0`
- demand `~ Poisson(5)`

Those are the nearest literature DRL settings for the single-item vanilla lost-sales problem in
this repo.

The broader `invman` vanilla benchmark grid is larger:

- lead times `L in {2, 4, 6, 8, 10}`
- shortage costs `p in {4, 19}`
- demand families `{Poisson, Geometric}`
- mean demand `5`

### Published neural architecture

Gijsbrechts et al. use the same A3C actor-critic backbone across lost sales, dual sourcing, and
multi-echelon:

- four fully connected layers with widths `[150, 120, 80, 20]`
- ReLU after each layer
- value regularization `0.25`
- four parallel learners
- gradient clipping `40`

The paper states that this backbone is fixed across the three problem types, and that only the
learning rate, entropy regularization, and buffer length are tuned by problem instance.

### Published action design and normalization

For lost sales, the paper uses a bounded scalar action space:

- actions `[0, 1, ..., 20]`
- companion-code parameter `max_order = 20`

The companion code also rescales the state by:

- `InvMax + LT * max_order`

Repo implication:

- `Q = 20` is the clean literature anchor if we want a bounded policy-side lost-sales baseline
- if a cap is used in `invman`, it should be explicit in the policy design rather than silently
  imposed by the environment

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

Paper-suite notes live in:

- [experiments/README.md](/home/nima/code/ml/invman/invman/problems/lost_sales/experiments/README.md)
