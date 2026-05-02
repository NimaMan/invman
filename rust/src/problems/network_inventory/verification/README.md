# Verification

`network_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- analytical reproduction of the single-node paper rows
- continuous paper-facing audit of the serial paper rows
- policy-state layout checks
- worked-transition accounting checks
- exact finite-horizon DP comparison against the carried pairwise base-stock heuristic

These checks are repo-native only. They validate the current paper-shaped discrete Rust
implementation on a tiny serial verifier and compare paper-facing benchmark rows against explicit
Rust reproductions. The single-node rows match exactly; the serial rows are still an audit layer,
not a literature-verified executable match.
