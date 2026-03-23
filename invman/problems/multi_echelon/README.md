# Multi Echelon

This package implements the two-echelon inventory system used by Gijsbrechts et al. (2022) after Van Roy et al. (1997):

- one warehouse
- `K` identical retailers
- warehouse lead time `l_w`
- retailer lead time `l_r`
- warehouse and retail capacity constraints
- hybrid lost-sales / same-day-expedite service with probability `P_w`

The learned-policy action matches the paper’s reduced action space:

- warehouse state-dependent base-stock level `y_w`
- retailer state-dependent base-stock level `y_r`

The v1 benchmark heuristic is the constant base-stock policy over the literature action grids.
