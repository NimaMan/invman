# Verification

`spare_parts_inventory` is exercised by executable assertions in `tests/verification.rs`.
This file states, honestly, which of those assertions are LITERATURE VERIFICATION (an
in-crate test re-runs an env/solver and reproduces a paper-printed number within a stated
tolerance) and which are not, per the repo rule in `rust/README.md`.

## Literature-verified (the only one in this family)

- `kranenburg_table_5_2_rows_are_reproduced_within_table_rounding`
  - re-runs the ANALYTICAL lateral-transshipment solver in
    `literature/kranenburg_lateral_transshipment.rs`
  - reproduces every printed row of Table 5.2 (Kranenburg 2006 PhD thesis, TU/e,
    Chapter 5, p.107) within table-rounding tolerance 0.02
  - this is a continuous-review, METRIC-style multi-location model, STRUCTURALLY DIFFERENT
    from the trainable `env.rs`; the verification covers the analytical module ONLY

## NOT literature verification (characterization / self-consistency / snapshot)

- `worked_transition_matches_expected_accounting`
  - pins one env.rs step against hand-computed accounting (drift guard)
- `env_periodic_review_trajectory_is_pinned_characterization_not_literature`
  - pins a full multi-period env.rs trajectory (actions, failures, per-period costs,
    deterministic repair return, total cost) against hand-computed env.rs accounting
- `exact_dp_dominates_repo_heuristics`
  - self-consistency only: the exact finite-horizon DP optimum dominates the carried
    heuristics on the reduced verification instance; reproduces no paper number
- `van_oers_2024_table_is_recorded_but_not_literature_verified`
  - frozen snapshot of the van Oers (2024) Table 1 constants; no env/solver re-runs them,
    so it is explicitly NOT verification
- `reference_set_has_expected_shape`, `raw_state_layout_matches_expected_shape`,
  `heuristic_first_actions_match_named_heuristic_evaluators`
  - structural / shape checks

## Flags

`references.rs` is the source of truth. It flags:

- Kranenburg Table 5.2 rows: `literature_verified = true`
- `PRIMARY_REFERENCE_INSTANCE` (env.rs canonical): `literature_verified = false`
- `VERIFICATION_PROBLEM_INSTANCE` (env.rs reduced DP instance): `literature_verified = false`
- van Oers (2024) scenarios: `literature_verified = false`
