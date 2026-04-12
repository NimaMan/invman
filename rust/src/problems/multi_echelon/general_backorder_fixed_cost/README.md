# General Backorder Fixed Cost

This subproblem carries the general-network CardBoard Company benchmark from Geevers et al. (2023).

Important scope note:

- the folder name stayed from the earlier split decision
- the carried Geevers benchmark does **not** include fixed ordering costs in the published objective
- the implemented objective is holding cost plus backorder cost, exactly as written in Section 3 of the
  CEJOR paper and Chapter 6 of the thesis

## Formulation

We model a two-echelon general network with:

- 4 suppliers with unlimited capacity
- 4 warehouses
- 5 retailers
- customer demand only at the 5 retailers
- backorders, not lost sales
- constant unit lead time on supplier-to-warehouse and warehouse-to-retailer shipments
- Poisson retailer demand with mean `15`

The CardBoard Company network uses the historical warehouse-to-retailer connection weights shown in
Figure 6.1 of the thesis:

- retailer 0: `(0.60, 0.30, 0.10)` from warehouses `(0, 1, 3)`
- retailer 1: `(0.50, 0.40, 0.10)` from warehouses `(0, 1, 3)`
- retailer 2: `(0.15, 0.80, 0.05)` from warehouses `(0, 1, 3)`
- retailer 3: `(0.10, 0.80, 0.10)` from warehouses `(1, 2, 3)`
- retailer 4: `(0.70, 0.30)` from warehouses `(2, 3)`

Per-period costs:

- warehouse holding cost: `0.6`
- retailer holding cost: `1.0`
- warehouse-to-retailer backorder cost at warehouses: `0.0`
- retailer-to-customer backorder cost: `19.0`

The Rust environment follows the benchmark simulator structure from the thesis appendix:

1. Receive incoming deliveries.
2. Suppliers ship current warehouse orders.
3. Warehouses fulfill current retailer orders, ranking retailers by retailer inventory minus retailer
   customer backorders when stock is short.
4. Retailers satisfy current customer demand and accumulate customer backorders.
5. Remaining inventory is used to clear existing warehouse-to-retailer and retailer-to-customer backorders.
6. New upstream orders are placed for the next period.

The environment exposes raw state only. Any feature normalization belongs in rollout or policy code.

## Literature Rows

Published benchmark source:

- Geevers et al. (2023), CEJOR 32:157-187
- DOI: `10.1007/s10100-023-00872-2`

Published benchmark rows from Table 7:

- set 1:
  - benchmark cost `10,467`
  - base-stock levels `[82, 100, 64, 83, 35, 35, 35, 35, 35]`
  - PPO best `8,714`
  - PPO average `630,401`
- set 2:
  - benchmark cost `4,797`
  - base-stock levels `[37, 47, 33, 63, 30, 30, 30, 30, 30]`
  - PPO best `4,175`
  - PPO average `314,923`
- set 3:
  - benchmark cost `4,797`
  - base-stock levels `[37, 47, 33, 63, 30, 30, 30, 30, 30]`
  - PPO best `3,935`
  - PPO average `4,481`

## Current Verification Status

This subproblem is **not literature-verified** yet.

Current Rust-side audit status:

- set 1 is close under the rewritten benchmark-faithful simulator
  - repo reproduced mean cost: about `10381.47`
  - published benchmark cost: `10467`
- set 2 and set 3 are still materially off the published benchmark cost under the most literal
  weighted-split interpretation
  - repo reproduced mean cost: about `15271.29`
  - published benchmark cost: `4797`

The likely source of the remaining gap is the open-paper ambiguity around how the set 2/3 benchmark
translates a 9-parameter base-stock policy into the paper's order-per-edge action setting. The public
paper says only that orders are "split across all upstream connections"; it does not specify the exact
benchmark conversion rule beyond that sentence.

So the honest benchmark status is:

- set 1: near-reproduced
- set 2 and set 3: carried as published rows, but not yet reproduced closely enough to claim
  literature verification
