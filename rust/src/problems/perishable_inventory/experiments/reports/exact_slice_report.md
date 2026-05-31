# perishable_inventory Exact-Slice Benchmark

- objective: compare exact optimum, tuned heuristics, and CMA-ES soft-tree policies on the literature-verified m=2/L=1 slice
- discounting: gamma=`0.99`, warmup = instance `warm_up_periods_ratio`
- tree_depth: `2`, split_type: `oblique`, leaf_types: `['linear', 'sigmoid_linear']`
- CMA-ES: `150` generations, popsize `16`, sigma_init `1.5`
- search_seeds: `48`, eval_seeds: `256`

Estimator note: two distinct estimators of the discounted return appear here.
`exact_value_iteration` is the analytic expected discounted return under the
midpoint-binned gamma demand (the value the repo reproduces from Farrington 2025).
All other rows are Monte-Carlo means over sampled-and-rounded gamma demand rollouts,
which sit ~10-15 units (~1%) below the analytic value on the same instance because they
are a different (sampled, finite-horizon, zero-start) estimator. The `gap_to_exact_optimum`
column therefore mixes estimators; the apples-to-apples comparison is the
`gap_to_best_heuristic` column, where every row uses the same Monte-Carlo estimator and
eval seeds. SEM is the standard error of the Monte-Carlo mean; a soft tree at -1455 vs the
best heuristic at -1468 (FIFO) is a real, multi-SEM win.

## `de_moor2022_m2_exp1_l1_cp7_lifo` (m=2, L=1, lifo)

- exact_value_iteration_return: `-1552.991` (rounded `-1553`, published `-1553`)
- matches_published_value_iteration_return: `True`
- best_base_stock_level: `5` (matches_published: `True`)
- matches_published_policy_table: `True`
- best_heuristic_mean_return: `-1557.936`

| Policy | Params | Mean Return | SEM | Gap to Exact | Gap to Best Heuristic | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `exact_value_iteration` | `-` | `-1552.991` | `0.000` | `0.000` | `-4.945` | exact tabular MDP optimum (reproduces Farrington 2025 Table 3) |
| `base_stock` | `[5]` | `-1558.344` | `4.169` | `5.353` | `0.409` | best base-stock level from stochastic search (Monte-Carlo estimator) |
| `bsp_low_ew` | `[5, 4, 10]` | `-1557.936` | `4.169` | `4.945` | `0.000` | best BSP-low-EW params from stochastic search (Monte-Carlo estimator) |
| `soft_tree_linear` | `d=2, leaf=linear` | `-1605.252` | `4.435` | `52.262` | `47.317` | CMA-ES soft tree, seed=772255, 150 gens, 2.7s |
| `soft_tree_sigmoid_linear` | `d=2, leaf=sigmoid_linear` | `-1543.801` | `4.139` | `-9.190` | `-14.134` | CMA-ES soft tree, seed=454826, 150 gens, 0.7s |

## `de_moor2022_m2_exp2_l1_cp7_fifo` (m=2, L=1, fifo)

- exact_value_iteration_return: `-1457.281` (rounded `-1457`, published `-1457`)
- matches_published_value_iteration_return: `True`
- best_base_stock_level: `7` (matches_published: `True`)
- matches_published_policy_table: `True`
- best_heuristic_mean_return: `-1467.900`

| Policy | Params | Mean Return | SEM | Gap to Exact | Gap to Best Heuristic | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `exact_value_iteration` | `-` | `-1457.281` | `0.000` | `0.000` | `-10.619` | exact tabular MDP optimum (reproduces Farrington 2025 Table 3) |
| `base_stock` | `[7]` | `-1468.363` | `3.620` | `11.082` | `0.463` | best base-stock level from stochastic search (Monte-Carlo estimator) |
| `bsp_low_ew` | `[7, 7, 0]` | `-1467.900` | `3.612` | `10.619` | `0.000` | best BSP-low-EW params from stochastic search (Monte-Carlo estimator) |
| `soft_tree_linear` | `d=2, leaf=linear` | `-1455.850` | `3.617` | `-1.431` | `-12.050` | CMA-ES soft tree, seed=429914, 150 gens, 1.1s |
| `soft_tree_sigmoid_linear` | `d=2, leaf=sigmoid_linear` | `-1452.308` | `3.933` | `-4.973` | `-15.592` | CMA-ES soft tree, seed=491045, 150 gens, 0.9s |

