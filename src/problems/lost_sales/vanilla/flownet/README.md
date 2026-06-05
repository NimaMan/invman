# lost_sales/flownet

This folder records the `lost_sales` family in the shared FlowNet language.

It stays separate from `env.rs`, `rollout.rs`, and the Python bindings so we can describe the
problem family cleanly before trying to generalize execution.

## The Seven Questions

1. What inventory states exist?

- on-hand inventory at a single stocking point
- an inbound lead-time pipeline for previously placed orders
- a customer-demand sink used for service accounting

2. How can material move or transform?

- a replenishment order is dispatched from an abstract supplier into the inbound pipeline
- one order reaches on-hand inventory each period when the pipeline advances
- on-hand inventory is consumed by customer demand

3. What random events occur?

- customer demand arrives each period
- the current Rust implementation supports Poisson, geometric, and two-state MMPP demand

4. What can the controller choose?

- one nonnegative replenishment quantity each period

5. What can the controller observe, and when?

- the current on-hand level
- the lead-time order vector
- in the current implementation the observed policy state is the pipeline vector with on-hand
  inventory folded into the first position

6. How is performance scored?

- procurement cost
- optional fixed order cost
- holding cost on ending on-hand inventory
- lost-sales penalty on unmet demand

7. What timing rules and constraints shape the system?

- the controller acts on the start-of-period pipeline state
- the oldest pipeline order is received into stock
- the new order is appended to the end of the pipeline
- demand is realized and served from available stock
- period costs are charged

Those answers are encoded in `formulation.rs`, `instance.rs`, and `verification/`.

## Verification

The `verification/` folder is where this formulation is checked against:

- structural FlowNet validity
- canonical event ordering assumptions
- executable heuristic reproduction for the standard lost-sales benchmark
- literature policy anchors for the policies we do not yet execute here

The first benchmark anchor is the repository's canonical `vanilla_l4_p4_poisson5` instance:

- optimal reference: `4.73`
- capped base stock: `4.80`
- myopic2: `4.82`
- myopic1: `5.06`
- svbs: `5.83`

Current executable verification covers:

- `myopic2`
- `myopic1`
- `svbs`
- rollout-backed learned policy measurements through the Rust soft-tree, linear, and neural
  rollout entrypoints

Current literature-only anchors are:

- `optimal_reference`
- `capped_base_stock`
