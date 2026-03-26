# Fixed-Order-Cost Lost Sales

This package contains the fixed setup-cost extension of the single-item lost-sales problem.

Canonical reference instance:

- name: `lit_pois_mu5_l4_p4_k5`
- lead time `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- holding cost `h=1`
- demand `~ Poisson(5)`

Reference grid used for the first literature-aligned benchmark subset:

- name: `literature_subset_poisson_mu5`
- `L in {1, 2, 3, 4}`
- `p in {4, 19}`
- `K in {5, 25}`
- demand `~ Poisson(5)`

The grid is derived from the benchmark family of Bijvank, Bhulai, and Huh (2015). The literature
provides the family and the heuristic classes, but not a clean exact per-instance cost table for
the canonical instance above, so the benchmark numbers in this repo are repo-native.

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
| `linear_gated_ordinal_quantity` | `8.76878` | trusted |
| `nn_categorical_quantity` | `10.27299` | provisional |
| `nn_gated_ordinal_quantity` | `8.73282` | trusted |
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

- [reference_instances.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/reference_instances.py)
- [heuristics.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/heuristics.py)
- [benchmark.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/benchmark.py)
- [env.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/env.py)

Detailed benchmark notes live in
[fixed_cost_l4_refresh.md](/Users/nimamanaf/Desktop/code/ML/inventory_management/docs/benchmarks/fixed_cost_l4_refresh.md).
