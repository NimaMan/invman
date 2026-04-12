# Paper Benchmark

Paper objective for this family:

- train policy classes for fixed-cost lost sales
- compare them against the standard parametric replenishment heuristics
- compare them against the exact optimal policy on a small literature-verified instance

Reported slices should follow two levels.

1. exact literature slice
   - `bijvank2015_table1_l2_p14_k5`
2. paper benchmark grid
   - fixed-cost extension of the lost-sales full grid
   - demand families `{Poisson, Geometric, MMPP2 positive, MMPP2 negative}`
   - `p in {4, 19}`
   - `K in {5, 25}`
   - `L in {2, 4, 6, 8, 10}`

Reason:

- the Table 1 instance gives a clean exact comparator with published heuristic rows
- the larger grid mirrors the vanilla lost-sales paper surface with the extra setup-cost axis

Heuristic comparators:

- `(s,S)`
- `(s,nQ)`
- modified `(s,S,q)`

Exact comparator:

- average-cost value iteration from `exact_value_iteration.rs`

Current status:

- the exact literature slice is ready
- the paper benchmark grid is now defined in `experiments/mod.rs` on the Rust side
- Python can retrieve it through:
  - `invman_rust.lost_sales_fixed_order_cost_list_experiment_grids()`
  - `invman_rust.lost_sales_fixed_order_cost_get_experiment_grid(name)`
  - `invman_rust.lost_sales_fixed_order_cost_expand_experiment_grid(name)`
