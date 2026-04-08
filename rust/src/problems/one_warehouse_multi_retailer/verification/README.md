# Verification

`one_warehouse_multi_retailer` is verified by executable assertions in `tests/verification.rs`.

Current verifier scope:

- reference-shape checks from `references.rs`
- policy-state layout checks
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against allocation heuristics

The frozen reference numbers asserted in tests are repo-native exact values on the reduced
verification instance.

This reduced verifier is not literature-verified; it exists to anchor the exact DP, the
allocation rules, and the carried echelon base-stock heuristic implementation.
