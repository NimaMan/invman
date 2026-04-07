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
