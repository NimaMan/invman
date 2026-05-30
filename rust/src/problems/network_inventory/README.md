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

Simulation (sim) finding:

- `serial_echelon_simulation.rs` drives the discrete `env.rs` simulator with the optimal ECHELON
  base-stock policy (not the carried installation/pairwise base-stock) to test whether the env
  reproduces the analytical optimum. It does not, and the gap is structural, not a tuning artifact:
  - at the analytical echelon levels [15, 9, 26] the env averages ~147 with a large backorder
    component (~75), because each node has a raw->finished PRODUCTION step in addition to the
    inter-stage shipping lead time, lengthening the effective lead time (~7 periods vs the 4
    transit periods Clark-Scarf models), and the env charges holding on outgoing in-transit
    pipeline inventory that the optimized Clark-Scarf cost treats as a constant
  - even at the env's own best echelon levels the simulated optimum is >100, well above 72.04
- conclusion: the published serial optimum is verified by the exact solver; `env.rs` is the
  richer Pirhooshyaran network model and is intentionally not the vehicle for reproducing the
  textbook Clark-Scarf optimum. The test `env_does_not_reproduce_clark_scarf_optimum_structural_gap`
  records this quantitatively.

Package layout:

- literature references: `literature/references.rs`
- verification code: `verification/tests.rs`
- exact small solver: `finite_horizon_dp.rs`
- exact serial Clark-Scarf optimizer: `clark_scarf_serial_exact.rs`
- echelon base-stock policy + env simulation: `serial_echelon_simulation.rs`
- heuristics: `heuristics/`
- rollout path: `rollout.rs`
- FlowNet projection: `flownet/`
- practical notes: `practical/`
- experiment notes: `experiments/`
