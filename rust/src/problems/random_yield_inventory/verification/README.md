# Verification

`random_yield_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- policy-state layout checks
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against the carried heuristics

This verifier is repo-native. It checks implementation correctness on the reduced exact instance,
but it is not a literature-verification claim.

The reduced exact verifier stores only the problem instance in the literature catalog. Optimal,
heuristic, and worked-transition accounting values are generated or asserted in verification code, not
stored as literature reference numbers.
