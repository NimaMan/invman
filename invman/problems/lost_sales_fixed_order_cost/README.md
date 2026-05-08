# Fixed-Order-Cost Lost Sales

This package contains the fixed setup-cost extension of the single-item lost-sales problem.

## Literature guidance

### Primary fixed-cost benchmark reference

- Marco Bijvank, Sandjai Bhulai, and Woonghee Tim Huh, *Parametric replenishment policies for
  inventory systems with lost sales and fixed order cost*, European Journal of Operational
  Research, 2015.
- DOI: <https://doi.org/10.1016/j.ejor.2014.11.036>
- Repo literature note: `docs/literature/fixed_order_cost_literature.md`

This paper is the core literature anchor for the fixed-order-cost lost-sales problem in this repo.

### What the fixed-cost literature gives us

The Bijvank et al. benchmark family gives:

- the fixed-cost lost-sales problem family
- the benchmark heuristic classes
- validation anchors for published small instances

The main heuristic classes used there are:

- `(s,S)`
- `(s,nQ)`
- modified `(s,S,q)`

### What it does not give us

This literature does not give:

- a deep-RL benchmark for fixed-cost lost sales
- a prescribed neural architecture
- a prescribed neural action head

So for learning-based fixed-cost experiments, this package necessarily combines:

- fixed-cost benchmark structure from Bijvank et al.
- learning-architecture guidance from nearby single-item DRL papers

### Nearest DRL architecture anchor

The closest single-item DRL reference we currently have is the vanilla lost-sales A3C study of
Gijsbrechts et al. (2022):

- DOI: <https://doi.org/10.1287/msom.2021.1064>
- four fully connected layers `[150, 120, 80, 20]`
- ReLU after each layer
- value regularization `0.25`
- four parallel learners
- gradient clipping `40`
- bounded scalar action space `[0, 1, ..., 20]` for their lost-sales setting

Repo implication:

- if we want a literature-style bounded NN baseline for fixed-order-cost lost sales, the cleanest
  imported cap is `Q = 20`
- if we use a larger cap such as `Q = 50`, that is a repo-specific design choice rather than a
  direct literature DRL anchor
- any such cap should remain a policy parameter, not an environment-side hidden restriction

Canonical reference instance:

- name: `lit_pois_mu5_l4_p4_k5`
- lead time `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- holding cost `h=1`
- demand `~ Poisson(5)`

Full benchmark grid used for the fixed-cost paper suite:

- name: `lost_sales_style_full_grid_mu5`
- `L in {4, 6, 8, 10}`
- `p in {4, 19}`
- `K in {5, 25}`
- demand families `{Poisson, Geometric, MMPP2 positive, MMPP2 negative}`
- mean demand `5`

Legacy literature-aligned subset kept for smaller Poisson-only runs:

- name: `literature_subset_poisson_mu5`
- `L in {1, 2, 3, 4}`
- `p in {4, 19}`
- `K in {5, 25}`
- demand `~ Poisson(5)`

The grid is derived from the benchmark family of Bijvank, Bhulai, and Huh (2015). The literature
provides the family and the heuristic classes, but not a clean exact per-instance cost table for
the canonical instance above, so the benchmark numbers in this repo are repo-native.

Published validation instance:

- name: `bijvank2015_table1_l2_p14_k5`
- lead time `L=2`
- shortage cost `p=14`
- fixed ordering cost `K=5`
- holding cost `h=1`
- demand `~ Poisson(5)`
- published references:
  - optimal cost `11.46`
  - best `(s,S)` at `s=17, S=23`, cost `11.62`
  - best `(s,nQ)` at `s=17, q=7`, cost `11.56`
  - best modified `(s,S,q)` at `s=17, S=23, q=7`, cost `11.50`

## Heuristic benchmark anchors

Current canonical long-run heuristic anchors (`1,000,000` periods, `10` seeds, `20%` warm-up):

| Heuristic | Parameters | Mean cost |
| --- | --- | ---: |
| `s,S` | `s=21, S=27` | `9.37145` |
| `s,nQ` | `s=22, q=8` | `9.18096` |
| modified `s,S,q` | `s=22, S=30, q=8` | `9.17436` |

## Learned policy benchmark anchors

Current paper-like benchmark suite on the same canonical instance:

| Policy family | Mean cost | Status |
| --- | ---: | --- |
| `linear_categorical_quantity` | `10.27299` | trusted |
| `linear_soft_gated_ordinal_quantity` | `8.76878` | trusted |
| `nn_categorical_quantity` | `10.27299` | provisional |
| `nn_soft_gated_ordinal_quantity` | `8.73282` | trusted |
| `soft_tree_depth2_linear_leaf` | `8.77418` | trusted |
| `soft_tree_depth1_linear_leaf` | `8.77846` | trusted |

Important note:

- `nn_categorical_quantity` currently matches the linear categorical baseline exactly and should be
  re-verified before publication claims rely on it

## Current conclusion

For this problem, the policy head matters more than the backbone:

- categorical quantity heads are poor on the fixed-cost variant
- gated ordinal quantity heads are materially better
- once the head is improved, even the linear policy becomes competitive with the tree variants

Supporting files:

- `reference_instances.py`
- `heuristics.py`
- `benchmark.py`
- `env.py`

Detailed benchmark notes live in `docs/benchmarks/fixed_cost_l4_refresh.md`.

Paper-suite notes live in:

- [experiments/README.md](/home/nima/code/ml/invman/invman/problems/lost_sales_fixed_order_cost/experiments/README.md)
