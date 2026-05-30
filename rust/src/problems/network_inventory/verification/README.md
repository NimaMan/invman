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

Remaining (sim) task:

- the discrete `env.rs` network simulator does not yet reproduce the serial analytical costs under
  a base-stock policy. The exact-theory anchor above verifies the published serial optimum; the
  env-simulation reproduction is tracked separately.

Repo-native worked-transition expected values are kept in `verification/fixtures.rs`, not in
`literature/references.rs`.
