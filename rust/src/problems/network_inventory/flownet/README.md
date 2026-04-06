# network_inventory/flownet

This folder records the `network_inventory` family in the shared FlowNet language.

It stays separate from `env.rs`, `rollout.rs`, and the Python bindings so we can describe the
graph-generalized inventory physics cleanly and then check that the formulation matches the
existing Rust implementation.

## The Seven Questions

1. What inventory states exist?

- graph-indexed on-hand inventory at stocking nodes
- graph-indexed backlog carried at demand nodes
- directed edge pipelines for in-transit shipments
- source-supply nodes that can inject flow into the network without on-hand depletion
- a demand sink used to describe service and backorder semantics

2. How can material move or transform?

- shipments are requested on directed edges
- the oldest shipment on each edge pipeline is received first
- source nodes dispatch requested shipments without depleting local on-hand stock
- non-source nodes dispatch from local on-hand stock and allocate proportionally if outgoing
  requests exceed available inventory
- on-hand inventory serves both old backlog and newly realized demand
- unmet demand is retained as backlog

3. What random events occur?

- each node has its own demand process
- the current Rust implementation supports deterministic and Poisson demand at each node

4. What can the controller choose?

- one nonnegative shipment-request vector over all directed edges

5. What can the controller observe, and when?

- graph-indexed on-hand inventory
- graph-indexed backlog
- inbound pipeline totals by node
- in-transit totals by directed edge
- node demand means
- remaining-horizon fraction
- in the current implementation the controller observes this pre-receipt state, then the period
  transition receives in-transit shipments before dispatching new ones

6. How is performance scored?

- holding cost on ending on-hand inventory by node
- backlog cost on ending backlog by node

7. What timing rules and constraints shape the system?

- the controller chooses edge shipment requests from the start-of-period state
- existing edge receipts arrive first
- new shipments are dispatched onto directed edge pipelines
- source nodes fulfill their full requests; non-source nodes are inventory-limited
- backlog and new demand are served from post-dispatch on-hand inventory
- unmet demand is carried forward as backlog
- holding and backlog costs are charged on the ending state

Those answers are encoded in `formulation.rs`, `instance.rs`, and `verification/`.

## Verification

The `verification/` folder checks four things:

- the FlowNet formulation is structurally valid
- the primary diamond-network reference instance maps cleanly into a FlowNet instance
- the worked transition reference matches the current Rust `step_state` accounting
- the exact-verification freeze keeps the expected optimal and node-base-stock discounted costs
