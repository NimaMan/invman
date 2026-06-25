# Multi-Echelon Tree Smoke Result

Primary instance:

- Van Roy / Gijsbrechts setting 2
- `l_w = 5`
- `l_r = 3`
- `K = 10`
- `mu = 0`
- `sigma = 20`
- `h_w = 3`
- `h_r = 3`
- `c_w = 0`
- `p = 60`
- `P_w = 0.8`
- `C_m = 100`
- `C_w = 1000`
- `C_r = 100`

Learned policy run:

- result file: `invman/outputs/results/multi_echelon_soft_tree_setting2_smoke.json`
- policy family: soft tree
- split type: oblique
- depth: `2`
- leaf type: linear
- training budget: `80` CMA-ES iterations, population `8`
- evaluation summary:
  - learned tree: `3776.45`

Constant base-stock benchmark on the same run:

- best constant base-stock policy on the setting-2 action grid: `3776.45`
- best constant policy parameters: `y_w = 100`, `y_r = 0`

Interpretation:

- the multi-echelon package, constant base-stock search, and Rust rollout path are working;
- the first soft-tree smoke run matched the best constant base-stock policy on the configured action grid.
