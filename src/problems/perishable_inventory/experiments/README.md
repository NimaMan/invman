# Paper Benchmark

Paper objective for this family:

- design policy classes for perishable inventory
- optimize their parameters with CMA-ES
- compare them against:
  - problem heuristics
  - the exact optimal policy when available

## Reported Instances

Use two reported slices.

1. exact small-state slice
   - `de_moor2022_m2_exp1_l1_cp7_lifo`
   - `de_moor2022_m2_exp2_l1_cp7_fifo`
2. medium practical slice
   - `de_moor2022_m4_exp6_l2_cp7_fifo`

Reason:

- the `m = 2` slice gives exact optimal comparators through value iteration
- the `m = 4` slice is more representative for learned-policy comparison and practical reporting

## Learned Policy Families

Initial reported learned-policy family:

- `soft_tree`
  - depth `2`
  - leaf types:
    - `linear`
    - `sigmoid_linear`

Future structured policy families can be added later, but the first paper benchmark should stay
simple and stable.

## Heuristic Comparators

Report against:

- `base_stock`
- `bsp_low_ew`

## Exact / Optimal Comparator

Available for the exact small-state slice:

- exact optimal policy from `value_iteration_mdp.rs`

Not used for the medium practical slice:

- report the heuristic comparison only

## Reported Metrics

Exact small-state slice:

- expected discounted return
- gap to exact optimum
- gap to best heuristic

Medium practical slice:

- mean period cost
- fill rate
- cycle service level
- waste rate
- mean holding inventory

## Report Table Intent

The paper table for this family should show:

1. on exact small instances, whether CMA-ES can recover or approach the optimal policy gap
2. on a larger practical instance, whether the designed policy beats the benchmark heuristics on
   the waste-service-cost tradeoff

## Executable Benchmark

Working runner (self-contained, no Rust rebuild):

- `scripts/perishable_inventory/run_exact_slice_benchmark.py`

Default outputs (pass `--output_json` / `--output_markdown`):

- `src/problems/perishable_inventory/experiments/reports/exact_slice_report.json`
- `src/problems/perishable_inventory/experiments/reports/exact_slice_report/README.md`

It exercises exactly the comparison above (exact optimum vs tuned `base_stock` /
`bsp_low_ew` vs CMA-ES soft tree) on the two `m = 2` exact instances, using the
installed `invman_rust` bindings and the current `invman.policy.Policy` /
`invman.cmaes.CMAES` API.

Deprecated runner:

- `scripts/perishable_inventory/run_paper_benchmark.py` (and its `common.py`) are
  currently non-executable: they import `invman.policies.soft_tree.SoftTreePolicy`,
  a module path removed from the installed `invman` package (current API is
  `invman.policy.Policy` with `backbone="soft_tree"`). This is a repo-wide drift
  affecting seven sibling-problem `common.py` files; fixing it touches the shared
  scripts layer and is out of scope here. The new runner above is the replacement.
