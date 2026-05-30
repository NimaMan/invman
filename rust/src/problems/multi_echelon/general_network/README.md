# general_network

Rust-first problem home for `general_network`.

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

This family is the **Pirhooshyaran & Snyder (2021) general supply-network model**, not the
textbook serial system. The clean textbook serial (Clark-Scarf) model lives in its own
literature-verified family, `multi_echelon/serial`.

Current status:

- the `env.rs` network model is **not** literature-verified yet
- single-node newsvendor rows are reproduced analytically (closed form)
- the serial benchmark optima this family carries (Pirhooshyaran Tables 2-3) are the TEXTBOOK
  Clark-Scarf optima, verified in the `multi_echelon/serial` family (exact + env simulation), not here

Why the env is not literature-verified here:

- `serial_echelon_simulation.rs` drives this `env.rs` with the optimal ECHELON base-stock policy
  and shows it does NOT reproduce the textbook serial optimum (72.04 for the Poisson 3-stage
  instance), and the gap is structural, not a tuning artifact:
  - at the analytical echelon levels [15, 9, 26] the env averages ~147 with a large backorder
    component (~75), because each node has a raw->finished PRODUCTION step in addition to the
    inter-stage shipping lead time, lengthening the effective lead time (~7 periods vs the 4
    transit periods Clark-Scarf models), and the env charges holding on outgoing in-transit
    pipeline inventory that the optimized Clark-Scarf cost treats as a constant
  - even at the env's own best echelon levels the simulated optimum is >100, well above 72.04
- the test `env_does_not_reproduce_clark_scarf_optimum_structural_gap` records this quantitatively
- this is expected: a richer model is not the simple problem. Reproducing it requires either
  Pirhooshyaran's own published numbers for this richer model (their serial simulation protocol
  could not be recovered from public sources), or using the `multi_echelon/serial` family for the
  textbook serial problem.

Package layout:

- literature references: `literature/references.rs`
- verification code: `verification/tests.rs`
- exact small solver: `finite_horizon_dp.rs`
- echelon base-stock policy + env simulation (structural-gap evidence): `serial_echelon_simulation.rs`
- heuristics: `heuristics/`
- rollout path: `rollout.rs`
- FlowNet projection: `flownet/`
- practical notes: `practical/`
- experiment notes: `experiments/`
