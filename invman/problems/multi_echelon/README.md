# Multi Echelon

This package implements the two-echelon inventory system used by Gijsbrechts et al. (2022) after Van Roy et al. (1997):

## Literature guidance

### Primary references

- Joren Gijsbrechts, Robert N. Boute, Jan A. Van Mieghem, and Dennis J. Zhang,
  *Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual
  Sourcing, and Multi-Echelon Problems*, Manufacturing & Service Operations Management, 2022.
- DOI: <https://doi.org/10.1287/msom.2021.1064>
- Dimitri P. Bertsekas, Benjamin Van Roy, Yucheng Lee, and John N. Tsitsiklis,
  *A Neuro-Dynamic Programming Approach to Retailer Inventory Management*, CDC 1997.
- DOI: <https://doi.org/10.1109/CDC.1997.652501>

### Published problem structure

- one warehouse
- `K` identical retailers
- warehouse lead time `l_w`
- retailer lead time `l_r`
- warehouse and retail capacity constraints
- hybrid lost-sales / same-day-expedite service with probability `P_w`

### Repo benchmark settings

The package currently transcribes the two settings used by Gijsbrechts et al. for the Van Roy
benchmark family.

Setting 1:

- name: `multi_echelon_setting1`
- warehouse lead time `2`
- retailer lead time `2`
- number of retailers `10`
- warehouse holding cost `3`
- retailer holding cost `3`
- warehouse expedited cost `0`
- warehouse lost-sale cost `60`
- expedite service probability `0.8`
- warehouse capacity `100`
- warehouse inventory cap `1000`
- retailer inventory cap `100`
- aggregate demand mean `5`
- aggregate demand std `14`
- warehouse base-stock grid `{50, 60, 70, 80, 90, 100}`
- retailer base-stock grid `{0, 5, 10, 15, 20, 25, 30, 35, 40}`

Setting 2:

- name: `multi_echelon_setting2`
- warehouse lead time `5`
- retailer lead time `3`
- number of retailers `10`
- warehouse holding cost `3`
- retailer holding cost `3`
- warehouse expedited cost `0`
- warehouse lost-sale cost `60`
- expedite service probability `0.8`
- warehouse capacity `100`
- warehouse inventory cap `1000`
- retailer inventory cap `100`
- aggregate demand mean `0`
- aggregate demand std `20`
- warehouse base-stock grid `{40, 50, 60, 70, 80, 90, 100}`
- retailer base-stock grid `{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`

The published comparisons are against:

- constant base-stock
- Van Roy neuro-dynamic programming
- A3C

Published claims recorded in `reference_instances.py`:

- setting 1 A3C improvement vs constant base-stock: about `9%`
- setting 2 A3C improvement vs constant base-stock: about `12%`
- Van Roy reported savings: about `10%`

The paper reports improvement percentages, not a clean exact per-setting table of total costs.

### Published neural architecture

The paper uses the same fixed A3C backbone here as in lost sales and dual sourcing:

- four fully connected layers `[150, 120, 80, 20]`
- ReLU after each layer
- value regularization `0.25`
- four parallel learners
- gradient clipping `40`

### Published action design

The learned-policy action matches the paper’s reduced action space:

- warehouse state-dependent base-stock level `y_w`
- retailer state-dependent base-stock level `y_r`

The action grid used by the paper is the Van Roy grid:

- warehouse base-stock choices `{50, 60, ..., 100}`
- retailer base-stock choices:
  - setting 1 `{0, 5, 10, ..., 40}`
  - setting 2 `{0, 5, 10, ..., 50}`

The paper also fixes the buffer size at `100` for the multi-echelon experiments.

Repo implication:

- multi-echelon already has a policy-side bounded discrete action grid
- there is no meaningful single scalar `Q` to import from lost sales

## Current repo action parameterization

The learned-policy action matches the paper’s reduced action space:

- warehouse state-dependent base-stock level `y_w`
- retailer state-dependent base-stock level `y_r`

The v1 benchmark heuristic is the constant base-stock policy over the literature action grids.
