# production_assembly_distribution_network

Rust-first problem home for `production_assembly_distribution_network`.

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

Current status (re-investigated 2026-05; see "Verification status" below):

- the `env.rs` per-period transition and cost are **faithful to the Pirhooshyaran & Snyder (2021)
  MDP** (their eq. 1-13 sequence of events and eq. 3 cost). This was checked equation-by-equation
  against the paper PDF (arXiv:2006.05608, Sec. 3.1-3.2) and by impulse/worked-transition tracing.
- the env is **not yet anchored to a published benchmark NUMBER** (no reproduced cost), so it is
  recorded as literature_verified = **no** (faithful model, missing published anchor)
- single-node newsvendor rows are reproduced analytically (closed form); the env ALSO reproduces
  the L=1 single-node newsvendor cost by simulation to ~0.3 (12.71 published vs ~13 simulated,
  residual = integer order/demand rounding)
- the serial benchmark optima this family carries (Pirhooshyaran Tables 2-3) are the TEXTBOOK
  Clark-Scarf optima, verified in the `multi_echelon/serial` family (exact + env simulation), not here

Verification status / why no published serial anchor yet:

- CORRECTION (prior README was wrong on the mechanism): the env does NOT add a per-node production
  delay. The paper states "The processing time to convert raw materials to finished goods at a given
  node is assumed to be zero," and `env.rs` implements exactly that -- an impulse order placed at the
  source arrives after exactly its shipment lead time and is converted to finished goods in the same
  period (verified by `scripts/.../reproduce_pirhooshyaran_serial_case3.py` tracing). The effective
  source->customer lead time for serial case 3 is exactly 2+1+1 = 4 periods, matching Clark-Scarf,
  NOT the ~7 the prior README claimed.
- CORRECTION: charging holding on outgoing in-transit pipeline inventory is FAITHFUL to the paper
  (eq. 3 explicitly counts in-transit inventory in h_ij), not a model deviation.
- ACTUAL root cause that the env does not reproduce 72.04 / 47.65 when driven with the carried
  Clark-Scarf OUL levels: a POLICY/LEVEL-INTERPRETATION mismatch. The carried levels
  (10.69, 5.53, 6.49) are Snyder & Shen ECHELON base-stock levels; Pirhooshyaran's pairwise policy
  (eq. 5) targets the LOCAL raw-material inventory position, which EXCLUDES finished goods. Because
  each node processes ALL its raw material on arrival (eq. 2), over-produced finished goods that a
  node cannot ship downstream accumulate INVISIBLY to the local position, causing oscillatory
  over-ordering and a growing finished stockpile (traced: at OUL [5,6,11] the env reaches ~102/period
  vs the paper's 47.65, dominated by backorder cost). Echelon levels are simply not the matching
  LOCAL targets.
- the test `env_does_not_reproduce_clark_scarf_optimum_structural_gap` records the ~147 echelon-level
  cost quantitatively (assertions verified by an independent Python re-simulation: cost ~147,
  backorder ~75).
- What remains to anchor a published number (see `next steps` in `verification/README.md`):
  recover Pirhooshyaran's exact OUL->position protocol for the pairwise simulation (the levels alone
  are insufficient), OR compute the correct LOCAL base-stock levels for this env and a published
  reference cost. The `multi_echelon/serial` family remains the env-verified home for the TEXTBOOK
  serial optimum.

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
