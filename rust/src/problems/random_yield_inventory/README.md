# random_yield_inventory

Rust-first problem home for `random_yield_inventory`.

## Formulation

Repo interpretation:

- single-item inventory with stochastic supply yield
- positive lead time
- heuristic and exact reduced benchmark support on discrete instances
- all-or-nothing arrival success in the repo-native executable model

The state consists of current inventory, the outstanding pipeline, and period index. The action is
the order quantity. Demand is stochastic, supply arrivals are stochastic through the yield-success
process, and the objective is discounted procurement plus holding and shortage cost.

Code lives under `rust/src/problems/random_yield_inventory/`.

Literature and verification anchors live in:

- `literature/references.rs`
- `verification/tests.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

Current status: not literature-verified.

Reason:

- Yan et al. (2026) is the closest exact model match, but the accessible record does not expose a
  public row-level benchmark table that the repo can assert against.
- Chen et al. (2018) is the main weighted-newsvendor anchor, but we have not recovered public
  benchmark numbers from that source.
- Inderfurth and Kiesmuller (2015) do publish numeric results, but for related general random-yield
  models rather than the repo's executable all-or-nothing environment.

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps its derived feature map in `rollout.rs`
- any normalization or expectation-based encoding must stay outside the environment layer
