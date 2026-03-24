# Fixed-Order-Cost Lost Sales

This package keeps the fixed-order-cost extension of the single-item lost-sales problem.

Canonical reference instance:

- name: `lit_pois_mu5_l4_p4_k5`
- lead time `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- holding cost `h=1`
- demand `~ Poisson(5)`

Benchmark status:

- the literature gives the benchmark family and policy classes, but not a clean per-instance exact-cost
  table for this instance
- this repo therefore uses a repo-native canonical benchmark anchored to the published instance

Current long-run benchmark anchors on the canonical instance:

- `s,S`: `9.44401`
- `s,nQ`: `9.21664`
- modified `s,S,q`: `9.16537`

Policy function approximator anchors on the same instance:

| Approximator | Policy head / structure | Eval horizon | Mean cost |
| --- | --- | ---: | ---: |
| Linear | categorical quantity | `50000` | `10.42369` |
| NN | gated ordinal quantity | `50000` | `9.51636` |
| Soft tree | oblique depth-2, linear leaves | `1000000` | `8.81009` |
| Soft tree | oblique depth-1, linear leaves | `1000000` | `8.76576` |

Supporting files:

- [reference_instances.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/reference_instances.py)
- [heuristics.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/heuristics.py)
- [benchmark.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/benchmark.py)
- [env.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/invman/problems/lost_sales_fixed_order_cost/env.py)

Detailed benchmark notes live in
[fixed_cost_l4_refresh.md](/Users/nimamanaf/Desktop/code/ML/inventory_management/docs/benchmarks/fixed_cost_l4_refresh.md).
