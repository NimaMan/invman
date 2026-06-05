# Verification

`joint_replenishment` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- policy-state layout checks
- worked-transition accounting checks
- full-truckload action-feasibility checks
- exact reduced finite-horizon DP comparison against the carried heuristics (self-consistency)
- published literature anchor checks (`VANVUCHELEN_2020_FIGURE3_ANCHOR`):
  - `published_figure3_anchor_has_expected_shape`: the carried anchor points at setting 5, state
    `(5,0)`, optimal action `(0,6)`, heuristic action `(2,4)`, and both actions are exactly one FTL.
  - `env_reproduces_figure3_anchor_one_period_cost`: the env one-period accounting (Eq. 2 / Eq. 4)
    at the published optimal action `(0,6)` matches the paper's cost convention for a worked demand.

Two notions of verification (accurate in-crate scope):

- ENVIRONMENT literature-verification: model fidelity is literature-verified (env Eq. 1-4 match the
  paper) and the env's one-period cost at the published optimal action `q=(0,6)` for state `(5,0)` is
  asserted in-crate (= 90 for demand `(2,4)`). IMPORTANT: the in-crate tests assert the CARRIED anchor
  shape plus this one-period cost identity; they do NOT re-derive that `q=(0,6)` is the optimum. The
  full optimality reproduction uses an INFINITE-HORIZON value iteration (the paper's setting) run
  ONLY in `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`, which is outside this crate
  and not part of `cargo test`. So the optimal-action reproduction is faithful-but-external, not an
  in-crate assertion; the in-crate guarantee is env cost-accounting fidelity at the published action.
- REPO self-consistency: the reduced FINITE-horizon (4-period, discounted) DP comparator confirms the
  exact DP dominates the carried heuristics. This is not the paper's infinite-horizon average-cost
  setting and is not asserted against the published action; optimal and heuristic costs are generated
  by the Rust solver at verification time and are not stored as literature reference numbers.
