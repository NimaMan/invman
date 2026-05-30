# Verification

`network_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- analytical reproduction of the single-node paper rows
- exact Clark-Scarf reproduction of all ten serial benchmark optimal costs
- policy-state layout checks
- worked-transition accounting checks
- exact finite-horizon DP comparison against the carried pairwise base-stock heuristic

Literature-verified anchors:

- single-node newsvendor rows: reproduced exactly by the closed-form newsvendor in
  `verification/literature_benchmarks.rs`
- serial Clark-Scarf optimal costs: `serial_rows_reproduced_by_exact_clark_scarf_solver` asserts
  that `clark_scarf_serial_exact.rs` reproduces every published serial optimal cost
  (Pirhooshyaran and Snyder 2021, Tables 2-3; case 3 = Snyder and Shen Example 6.1, cost 47.65)
  within 0.5% relative error. The solver is cross-checked against Snyder's `stockpyl.ssm_serial`
  reference implementation, and on discrete Poisson instances reproduces it to machine precision.

Simulation (sim) investigation:

- `serial_echelon_simulation.rs` drives `env.rs` with the optimal ECHELON base-stock policy. The
  test `env_does_not_reproduce_clark_scarf_optimum_structural_gap` records that the env does not
  reproduce the analytical optimum (72.04): at the analytical levels it averages ~147 with a large
  backorder component, and even its own best levels exceed 100. The cause is structural -- the env
  adds a per-node raw->finished production step (lengthening the effective lead time) and charges
  holding on in-transit pipeline inventory that the optimized Clark-Scarf cost treats as constant.
- Therefore the published serial optimum is verified by the exact solver, and `env.rs` (the richer
  Pirhooshyaran network model) is documented as a different system rather than forced to match.

Repo-native worked-transition expected values are kept in `verification/fixtures.rs`, not in
`literature/references.rs`.
