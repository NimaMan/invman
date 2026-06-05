# Verification

`ameliorating_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- state-layout checks for policy features
- worked-transition accounting checks
- exact reduced finite-horizon DP comparison against carried heuristics

These checks are repo-native only. They validate the reduced Rust implementation, not published
paper rows. The worked-transition accounting was independently re-confirmed through the installed
`ameliorating_inventory_policy_rollout_from_paths` binding (one-period rollout reproduces the
documented `period_cost = -9.5`).

The package is NOT literature-verified: the env is a reduced approximation of Pahr and Grunow
(2025), not a faithful port. See `../literature/README.md` for the precise, term-by-term fidelity
gap and the recorded (non-anchoring) published upper-bound anchors.

The reduced exact verifier stores only the problem instance in the literature catalog. Optimal,
heuristic, and worked-transition accounting values are generated or asserted in verification code, not
stored as literature reference numbers.
