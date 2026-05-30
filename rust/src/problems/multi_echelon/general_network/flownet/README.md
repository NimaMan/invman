# general_network/flownet

This folder records the `general_network` family in the shared FlowNet language.

It stays separate from `env.rs`, `rollout.rs`, and the Python bindings so we can describe the
graph-generalized inventory physics cleanly and then check that the formulation matches the
existing Rust implementation.

## The Seven Questions

1. What inventory states exist?

- finished-goods inventory by node
- raw-material inventory by supply relation
- internal backorders by internal edge
- external backorders by node
- supply-relation pipelines for in-transit shipments, including external-supplier relations

2. How can material move or transform?

- each supply relation can receive a nonnegative request
- received items first enter relation-indexed raw-material inventory
- nodes immediately process raw material into finished goods
- `assembly_and` nodes consume one unit from each predecessor relation per finished unit
- `assembly_or` and `single` nodes consume all currently available raw material
- scarce finished goods are allocated proportionally across internal and external demand

3. What random events occur?

- each node has its own external demand process
- the current Rust implementation supports deterministic, Poisson, and Normal demand models

4. What can the controller choose?

- one nonnegative supply-request vector over all supply relations

5. What can the controller observe, and when?

- finished inventory by node
- raw inventory totals by node
- relation-indexed raw inventory and in-transit totals
- internal and external backlog
- node demand means
- current-period realized external demands
- remaining-horizon fraction
- the current implementation observes current-period external demand before choosing requests,
  matching the paper’s demand-then-order sequence

6. How is performance scored?

- linear holding cost on ending paper-style inventory totals
- linear backlog cost on ending internal and external backorders

7. What timing rules and constraints shape the system?

- external demand is realized first
- downstream nodes request supply before upstream ones
- receipts are processed into raw material, then into finished goods
- upstream nodes ship after requests are known
- internal and external shortages become backorders
- holding and backlog costs are charged on the ending state

Those answers are encoded in `formulation.rs`, `instance.rs`, and `verification/`.

## Verification

The `verification/` folder checks four things:

- the FlowNet formulation is structurally valid
- the primary serial reference instance maps cleanly into a FlowNet instance
- the worked transition reference matches the current Rust `step_state` accounting
- the exact-verification freeze keeps the expected optimal and pairwise-base-stock discounted costs
