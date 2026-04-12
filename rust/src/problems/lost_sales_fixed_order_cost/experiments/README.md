# Paper Benchmark

Paper objective for this family:

- train policy classes for fixed-cost lost sales
- compare them against the standard parametric replenishment heuristics
- compare them against the exact optimal policy on a small literature-verified instance

Reported slices should follow two levels.

1. exact literature slice
   - `bijvank2015_table1_l2_p14_k5`
2. medium practical slice
   - a larger repo-native fixed-cost lost-sales instance

Reason:

- the Table 1 instance gives a clean exact comparator with published heuristic rows
- the larger practical slice is where learned-policy comparisons become more meaningful

Heuristic comparators:

- `(s,S)`
- `(s,nQ)`
- modified `(s,S,q)`

Exact comparator:

- average-cost value iteration from `exact_value_iteration.rs`

Current status:

- the exact literature slice is ready
- the medium practical slice still needs to be defined and reported
