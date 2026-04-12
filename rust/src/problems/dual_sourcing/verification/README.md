# Verification

`dual_sourcing` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- the full six-row Figure 9 optimality-gap table frozen in tests
- worked-transition accounting checks on the reduced state update
- bounded-DP benchmark reproduction of the published gap labels on the canonical
  `dual_l2_ce105` verification instance

Current tolerance policy:

- exact reduced-state transition checks use exact equality
- the literature comparison uses a `0.01` percentage-point absolute tolerance on optimality gaps

For a full six-instance batch comparison against the carried literature labels, use the repo helper
`scripts/dual_sourcing/validate_reference_grid.py`. The larger `l_r in {3,4}` bounded-DP slices are
substantially heavier than the canonical unit-test instance, so they are tracked through that batch
validation path rather than the default Rust unit-test path.
