# network_inventory

Rust-first problem home for `network_inventory`.

Problem formulation in the source paper:

- finite-horizon stochastic multi-echelon inventory optimization on a directed acyclic network
- raw-material and finished-goods inventories, including assembly and distribution structures
- pairwise order-up-to decisions on supply relations
- downstream-to-upstream ordering after current-period demands are observed
- upstream-to-downstream receipt, processing, shipping, and cost assessment

Current Rust interpretation:

- paper-shaped discrete network model
- finished-goods inventory by node
- raw-material inventory by supply relation
- internal backorders by edge and external backorders by node
- supply pipelines by supply relation, including external-supplier relations
- `single`, `assembly_and`, and `assembly_or` node modes
- pairwise base-stock heuristic over supply relations
- repo-native exact verifier on a tiny serial network with explicit supplier and customer sides
- exact Clark-Scarf serial decomposition solver for the published serial optimal costs

Current status:

- serial Clark-Scarf optimal costs are literature-verified (exact-theory anchor)
- single-node paper rows are analytically reproducible
- env-simulation reproduction of the serial costs is the remaining (sim) task

What is literature-verified here:

- the single-node newsvendor rows are closed-form and reproduced exactly
- all ten serial benchmark rows (Pirhooshyaran and Snyder 2021, Tables 2-3) are the classical
  periodic-review serial multi-echelon optimum. Case 3 is Snyder and Shen "Fundamentals of Supply
  Chain Theory" Example 6.1 (optimal cost 47.65). The exact Clark-Scarf decomposition solver
  `clark_scarf_serial_exact.rs` reproduces every published optimal cost within 0.05% relative
  error, cross-checked against Snyder's public `stockpyl.ssm_serial` reference implementation

What is not yet literature-verified here:

- the discrete `env.rs` network simulator does not yet reproduce the serial analytical costs under
  a base-stock policy. That env-side simulation protocol is the open (sim) task and is separate
  from the exact-theory anchor above. The published serial values themselves are now verified
  against exact theory, not merely cataloged.

Package layout:

- literature references: `literature/references.rs`
- verification code: `verification/tests.rs`
- exact small solver: `finite_horizon_dp.rs`
- exact serial Clark-Scarf optimizer: `clark_scarf_serial_exact.rs`
- heuristics: `heuristics/`
- rollout path: `rollout.rs`
- FlowNet projection: `flownet/`
- practical notes: `practical/`
- experiment notes: `experiments/`
