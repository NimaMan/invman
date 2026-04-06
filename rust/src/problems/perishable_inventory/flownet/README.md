# perishable_inventory/flownet

This folder records the `perishable_inventory` family in the shared FlowNet language.

It stays separate from `env.rs`, `rollout.rs`, and the exact MDP implementation so we can describe
the family structure cleanly and check that the formulation matches the existing Rust semantics.

## The Seven Questions

1. What inventory states exist?

- age-structured on-hand inventory, ordered from youngest to oldest
- an inbound replenishment pipeline
- a customer-demand sink
- an expiration / waste sink for units that age out unsold

2. How can material move or transform?

- a replenishment order is dispatched from a supplier into the inbound pipeline
- one pipeline order arrives into the youngest age bucket when the pipeline advances
- remaining units age forward one bucket each period
- the oldest unsold units expire into waste
- on-hand inventory is issued to customer demand

3. What random events occur?

- customer demand is sampled each period
- the current Rust rollout uses rounded Gamma demand with configured mean and coefficient of
  variation

4. What can the controller choose?

- one nonnegative replenishment quantity each period

5. What can the controller observe, and when?

- the pipeline order vector
- the on-hand age-bucket vector
- the current Rust policy state exposes pipeline orders first, then on-hand age buckets

6. How is performance scored?

- procurement cost on the order quantity
- holding cost on non-expired remaining inventory
- lost-sales penalty on unmet demand
- waste cost on expired inventory

7. What timing rules and constraints shape the system?

- the controller acts on the current pipeline-plus-age-bucket state
- demand is realized and served before aging and expiration are applied
- leftover oldest inventory expires
- remaining inventory ages one bucket older
- the inbound pipeline advances and the arrival enters the youngest bucket

Those answers are encoded in `formulation.rs`, `instance.rs`, and `verification/`.

## Verification

The `verification/` folder checks four things:

- the FlowNet formulation is structurally valid
- the primary literature reference instance maps cleanly into a FlowNet instance
- FIFO and LIFO service semantics match the current Rust `step_state` implementation
- the primary literature reference keeps the expected optimal and best-base-stock discounted
  returns, with the correct best base-stock level
