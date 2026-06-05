# Verification

`one_warehouse_multi_retailer` is verified by executable assertions in `tests/verification.rs`.

Current verifier scope:

- reference-shape checks from `references.rs` (14 Table A.3 instances, primary = instance 7)
- policy-state layout checks (`raw_state_layout_matches_expected_shape`)
- worked-transition accounting checks (`worked_transition_matches_expected_accounting`) — a
  full lost-sales period traced by hand and asserted against the frozen reference
- proportional allocation exhausts available inventory
  (`proportional_allocation_uses_all_available_inventory_when_orders_exceed_supply`)
- symmetric echelon-target expansion (`symmetric_echelon_target_mode_expands_shared_retailer_target`)
- exact reduced finite-horizon DP dominance over the allocation heuristics
  (`finite_horizon_dp_dominates_repo_heuristics`): `optimal <= proportional` and
  `optimal <= min_shortage` on `VERIFICATION_PROBLEM_INSTANCE`

The frozen reference numbers asserted in tests are repo-native exact values on the reduced
verification instance. Reproduced live against the installed extension via
`one_warehouse_multi_retailer_exact_dp_summary()`: optimal `8.485`, proportional `9.2225`,
min_shortage `9.2225`.

## What this verifies vs what it does NOT

- **Verifies (transition + cost fidelity):** the env's order of events, holding/penalty
  accounting, pipeline advance, and the two allocation rules are internally consistent and are
  dominated by the true finite-horizon optimum. This is a correct-exact-solver check.
- **Does NOT verify against the paper's numbers:** `VERIFICATION_PROBLEM_INSTANCE` carries
  `literature_verified = false` (`references.rs:582`). It is a tractable repo-native anchor, not
  a Kaynov Table A.3 row. The reduced instance's optimal cost is not a published quantity.

This reduced verifier therefore anchors the **implementation**. The **literature** status is
`partial`: the 14 Table A.3 instance parameters and published rows are carried faithfully and
corroborated (the carried PPO gaps land in the paper's stated 1-3% lost-sales / 12-20%
partial-backorder bands), but the repo's heuristic costs reproduce the published costs only
approximately (~1-6%, regime-dependent sign). See `literature/README.md` for the cost-row table
and the root-cause discussion.
