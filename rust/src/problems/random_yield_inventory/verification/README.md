# Verification

`random_yield_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `references.rs`
- policy-state layout checks
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against the carried heuristics

This verifier is repo-native. It checks implementation correctness on the reduced exact instance,
but it is not a literature-verification claim.
