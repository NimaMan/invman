# Verification

`network_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- analytical reproduction of the single-node paper rows
- policy-state layout checks
- worked-transition accounting checks
- exact finite-horizon DP comparison against the carried pairwise base-stock heuristic

These checks are repo-native only. They validate the current paper-shaped discrete Rust
implementation on a tiny serial verifier and compare only the single-node paper rows against
explicit Rust reproductions. The serial paper rows are cataloged in `literature/` but are not in
verification because the public sources were insufficient to recover a stable executable protocol.

Repo-native worked-transition expected values are kept in `verification/fixtures.rs`, not in
`literature/references.rs`.
