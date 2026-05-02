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

Current status:

- not literature-verified
- repo-exact verified on the small serial verifier
- single-node paper rows are analytically reproducible
- serial paper rows are carried and audited, but not yet reproduced tightly enough

Reason:

- the package now follows the paper’s state and event structure much more closely
- the single-node newsvendor rows are closed-form and can be reproduced exactly
- the serial paper tables depend on a simulation protocol whose published recurrence is not fully
  self-consistent, so the current audit remains a best-effort paper-facing reproduction rather
  than a verified executable match

Package layout:

- literature references: `literature/references.rs`
- verification code: `verification/tests.rs`
- exact small solver: `finite_horizon_dp.rs`
- heuristics: `heuristics/`
- rollout path: `rollout.rs`
- FlowNet projection: `flownet/`
- practical notes: `practical/`
- experiment notes: `experiments/`
