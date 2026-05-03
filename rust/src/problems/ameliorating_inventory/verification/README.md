# Verification

`ameliorating_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- state-layout checks for policy features
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against carried heuristics

These checks are repo-native only. They validate the reduced Rust implementation, not published
paper rows.

The reduced exact verifier stores only the problem instance in the literature catalog. Optimal,
heuristic, and worked-transition accounting values are generated or asserted in verification code, not
stored as literature reference numbers.
