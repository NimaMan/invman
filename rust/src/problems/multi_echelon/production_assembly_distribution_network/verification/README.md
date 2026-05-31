# Verification

`production_assembly_distribution_network` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- analytical reproduction of the single-node paper rows
- exact Clark-Scarf reproduction of all ten serial benchmark optimal costs
- policy-state layout checks
- worked-transition accounting checks
- exact finite-horizon DP comparison against the carried pairwise base-stock heuristic

What is checked here (note: this env is NOT literature-verified):

- single-node newsvendor rows: reproduced exactly by the closed-form newsvendor in
  `verification/literature_benchmarks.rs` (analytical, not via the env simulation)
- serial-row consistency: `serial_rows_reproduced_by_exact_clark_scarf_solver` checks that the
  serial benchmark rows carried here (Pirhooshyaran and Snyder 2021, Tables 2-3; case 3 = Snyder
  and Shen Example 6.1, cost 47.65) equal the textbook Clark-Scarf optima, using
  `multi_echelon/serial::exact`. The serial optimum itself is literature-verified (exact + env
  simulation) in the `multi_echelon/serial` family, not here.

Simulation (sim) investigation (re-investigated 2026-05, corrected):

- env FIDELITY to the paper was re-checked equation-by-equation against arXiv:2006.05608, Sec.
  3.1-3.2: the per-period sequence (eq. 5-13), the "process all raw on arrival" production (eq. 1-2,
  6-7), the proportional allocation + backorder update (eq. 8-13), and the cost (eq. 3, which counts
  raw + finished + in-transit inventory in h_ij and is summed once per predecessor). `env.rs` matches
  all of these. The worked-transition fixture period cost (7.0) was re-derived by hand and matches.
- The env reproduces the paper's L=1 single-node newsvendor cost by SIMULATION (~13 vs published
  12.71; residual = integer order/demand rounding), confirming the core dynamics are correct.

- `serial_echelon_simulation.rs` drives `env.rs` with the Clark-Scarf ECHELON base-stock levels. The
  test `env_does_not_reproduce_clark_scarf_optimum_structural_gap` records cost ~147 with backorder
  ~75 (numbers independently reproduced by a stand-alone Python re-simulation of the env). PRIOR
  README CLAIM CORRECTED: this is NOT caused by a per-node production delay. The paper sets processing
  time to zero and the env does too; an impulse order placed at the source reaches finished goods
  after exactly its shipment lead time (verified), so the effective serial lead time is 4, matching
  Clark-Scarf. Charging holding on in-transit inventory is faithful to eq. 3, not a deviation.
- ACTUAL cause: a local-vs-echelon POLICY/LEVEL-INTERPRETATION mismatch. The carried OUL levels are
  ECHELON base-stock levels; Pirhooshyaran's pairwise policy (eq. 5) targets the LOCAL raw-material
  inventory position, which excludes finished goods. Because nodes process all raw on arrival,
  over-produced finished goods accumulate invisibly to the local position, causing oscillatory
  over-ordering. Driven with the carried levels rounded to [5,6,11] the env reaches ~102/period vs
  the paper's 47.65, dominated by backorder cost (traced).

What remains to make this env literature_verified (next steps):

1. Recover Pirhooshyaran's exact OUL->inventory-position protocol for their pairwise base-stock
   SIMULATION (Table 3 reports 47.65 from simulating THEIR env with the analytical OULs). The OUL
   levels alone are insufficient; the position definition and warm-start matter. With it, add an
   expected published serial cost (e.g. 47.65 for case 3) as a verification target and assert the env
   simulation reproduces it within ~2%.
2. Alternatively, compute the correct LOCAL base-stock levels for THIS env (e.g. by DFO / search over
   per-relation levels) and record both the levels and the resulting simulated cost as a self-consistent
   anchor, noting it is an env-native optimum rather than a paper-published number.
3. The reproduction harness is ready at
   `scripts/production_assembly_distribution_network/reproduce_pirhooshyaran_serial_case3.py`.

Repo-native worked-transition expected values are kept in `verification/fixtures.rs`, not in
`literature/references.rs`.
