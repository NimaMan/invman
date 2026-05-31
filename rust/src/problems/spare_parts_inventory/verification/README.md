# Verification

`spare_parts_inventory` is verified by executable assertions in `tests/verification.rs`.

Current verifier scope:

- reference-shape checks from `references.rs`
- policy-state layout checks
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against the carried heuristics
- exact analytical reproduction of Kranenburg (2006) Table 5.2 for the lateral-transshipment subfamily

The frozen reference numbers asserted in tests are repo-native exact values on the reduced
verification instance plus literature-verified Chapter 5 table values for the Kranenburg
subfamily.

Verification status (re-confirmed 2026-05-31 against the installed `invman_rust`):

- Kranenburg (2006) Table 5.2 is LITERATURE-VERIFIED: the analytical solver in
  `literature/kranenburg_lateral_transshipment.rs` reproduces all 35 published rows. The
  situation-1 randomized base-stock construction was independently re-derived from first
  principles (base case: `R*=9.09`, `C(R*)=91.90`, situation-3 `R*=6.10`, `C(R*)=63.00`,
  ratio 1.46) and matches the published numbers. Worst absolute deviation across all 35 rows
  is 0.005, against the 0.02 table-rounding tolerance.
- The reduced finite-horizon DP and the primary 17-period instance are repo-native and
  NOT verified against any published number; the bindings flag this explicitly as
  `repo_exact_solver_not_verified_against_literature`. The DP only anchors internal
  self-consistency (it must weakly dominate the carried heuristics).
- van Oers et al. (2024) Table 1 is carried as a table-only catalog (recorded exactly as
  published); no repo solver re-derives it. The 2026-05-31 literature audit verified the
  bibliographic metadata via Crossref (IFAC-PapersOnLine 58(19), 1006-1011,
  DOI 10.1016/j.ifacol.2024.09.144) but did not re-confirm the individual Table 1 cell
  values. The scenario structs set `literature_verified: true`, but for this block that
  flag means only "transcribed from a published table", not "reproduced by a solver".

Reproduce every verification block (without rebuilding Rust) via
`scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py`. The pre-existing helper
`scripts/spare_parts_inventory/common.py` was repaired on 2026-05-31: it imported a removed
`invman.policies.soft_tree.SoftTreePolicy`; it now builds the current `invman.policy.Policy`
descriptor so `train_soft_tree_reference.py`, `validate_against_exact_dp.py`, and
`validate_kranenburg_lateral_transshipment.py` import and run again.
