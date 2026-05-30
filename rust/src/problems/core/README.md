# problems/core

This folder is the conceptual backbone for inventory problems.

It is not another environment implementation and it is not a registry of named families.

The purpose of `problems/core` is to define the shared FlowNet language we use to describe any
inventory problem before we write a problem-specific `env.rs`.

## Fundamental Questions

Every inventory problem should answer the same small set of questions:

1. What inventory states exist?
2. How can material move or transform?
3. What random events occur?
4. What can the controller choose?
5. What can the controller observe, and when?
6. How is performance scored?
7. What timing rules and feasibility constraints shape the system?

These are the real modeling questions behind the repo's current problem families.

Examples:

- `lost_sales`
  - state: on-hand inventory plus inbound pipeline
  - movement: procurement arrival and customer demand
  - randomness: demand
  - control: order quantity
  - observation: full pipeline state
  - objective: holding + shortage + procurement
- `perishable_inventory`
  - adds age buckets and aging / waste mechanics to the physical layer
- `general_network`
  - expands the physical layer into a directed stock-flow network with node demand and edge
    pipelines
- disruption scenarios such as oil logistics with a closed Strait of Hormuz
  - can be represented by the same questions, with a richer physical network and a disruption
    process in the stochastic layer

## Layered Interpretation

We organize those questions into five layers.

Events cut across those layers and deserve their own first-class folder as the execution vocabulary
of the system.

### 1. Physical Layer

This is the base map of the system.

It defines:

- stock nodes
- pipelines
- flow edges
- material attributes

Typical examples:

- on-hand inventory
- source supply nodes
- backlog
- age buckets
- repair queues
- returnable stock
- shipments moving through a lead-time pipeline
- edge-indexed transit pipelines with edge-specific lead times

### 2. Stochastic Layer

This layer describes the shocks that hit the physical system.

Those shocks do not need to be estimated inside FlowNet.

For real systems, they can be supplied from outside as scenario inputs, stress paths, or externally
generated event streams.

It defines:

- demand
- yield
- failures
- returns
- forecast evolution
- disruptions
- transit delays

### 3. Control Layer

This layer describes what the decision-maker can do and what information they have.

FlowNet does not require an optimizer inside this layer.

It is enough to specify the admissible control regime and then simulate how the system evolves under
that supplied control.

It defines:

- action schema
- observation model
- service semantics
- feasibility constraints

Examples:

- scalar replenishment
- dual-source ordering
- shipment allocation
- removal decisions
- FIFO / LIFO issuance discipline
- lost-sales versus backlog service reaction
- local-information control

### 4. Objective Layer

This layer defines how the system is scored.

It defines:

- holding cost
- backlog or lost-sales penalty
- procurement cost
- fixed order cost
- waste cost
- salvage credit
- emergency fulfillment cost
- discounting

### 5. Timing Layer

This layer defines event order and operational feasibility.

It answers questions like:

- do receipts happen before demand?
- when does the controller act?
- when do aging, decay, repair, or removal happen?
- which capacities and other constraints bind each period?

Without this layer, two models with the same components can still behave differently.

## Event Vocabulary

The repo should treat events as first-class building blocks.

An action is not the same thing as an event, and an event is not the same thing as an accounting
term.

Examples:

- action: "order 10 units"
- event: "10 units were dispatched into the inbound pipeline"
- later event: "10 units were received"
- later event: "8 units of demand were served and 2 units were lost"
- accounting event: "holding cost and lost-sales penalty were charged"

The current event taxonomy is grouped into:

- exogenous events
  - demand arrival, failure occurrence, yield realization, disruption start/end
- control events
  - procurement, shipment, allocation, removal, reserve-release decisions
- material events
  - receipt, dispatch, delivery, transfer, repair completion
- transformation events
  - aging, decay, repair start, refinement, reclassification
- service events
  - demand served, backordered, lost, emergency fulfillment
- accounting events
  - holding cost, shortage penalty, procurement cost, waste cost, salvage credit

## Current Core Files

The current Rust skeleton now uses the folder structure directly:

```text
rust/src/problems/core/
  README.md
  mod.rs
  flownet/
    README.md
    mod.rs
    question.rs
    formulation.rs
    instance.rs
    validation.rs
  events/
    mod.rs
    kind.rs
    catalog.rs
    semantics.rs
  physical/
    mod.rs
    topology.rs
    stock.rs
    pipeline.rs
    flow.rs
    material.rs
  stochastic/
    mod.rs
    demand.rs
    process.rs
    yield.rs
    failure.rs
    return_flow.rs
    disruption.rs
    forecast.rs
  control/
    mod.rs
    action.rs
    observation.rs
    service.rs
    constraints.rs
  objective/
    mod.rs
    cost_term.rs
    reward.rs
    metrics.rs
    discounting.rs
  timing/
    mod.rs
    stage.rs
    scheduled_event.rs
    constraints.rs
    schedule.rs
```

Responsibilities:

- `flownet/`
  - the canonical stock-flow problem language
  - the fundamental question set
  - formulation, instance, and validation helpers
- `events/`
  - the typed event vocabulary used by schedules
  - event categories and named event specs
- `physical/`
  - stock nodes, pipelines, flow edges, material attributes, topology
- `stochastic/`
  - demand, yield, failure, return, disruption, and forecast process placeholders
- `control/`
  - actions, observations, service semantics, and feasibility constraints
- `objective/`
  - cost / reward terms and discounting
- `timing/`
  - event stages and schedule bindings
  - references to the event catalog

## Proposed Future Folder Structure

As this matures further, the likely structure is:

```text
rust/src/problems/core/
  README.md
  mod.rs
  flownet/
    README.md
    mod.rs
    question.rs
    formulation.rs
    instance.rs
    validation.rs
  events/
    mod.rs
    kind.rs
    catalog.rs
    semantics.rs
  physical/
    mod.rs
    topology.rs
    stock.rs
    pipeline.rs
    flow.rs
    material.rs
  stochastic/
    mod.rs
    demand.rs
    process.rs
    yield.rs
    failure.rs
    return_flow.rs
    disruption.rs
    forecast.rs
  control/
    mod.rs
    action.rs
    observation.rs
    service.rs
    constraints.rs
  objective/
    mod.rs
    cost_term.rs
    reward.rs
    metrics.rs
    discounting.rs
  timing/
    mod.rs
    stage.rs
    scheduled_event.rs
    constraints.rs
    schedule.rs
```

The important thing is that `core` is now centered on the fundamental modeling questions and the
layered FlowNet they imply, rather than on a hand-written taxonomy of the current families.
