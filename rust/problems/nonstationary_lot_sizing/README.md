# nonstationary_lot_sizing

Canonical Rust-first home for the nonstationary lot-sizing family.

Code:

- implementation: `rust/src/problems/nonstationary_lot_sizing/`
- tests: `rust/src/problems/nonstationary_lot_sizing/tests/verification.rs`

Artifact folders:

- `literature/`
  - paper scope and benchmark interpretation
- `practical/`
  - checked-in rolling forecast trace, benchmark spec, and latest report snapshot
- `experiments/`
  - paper-facing benchmark definition
- `verification/`
  - human-readable statement of what the verifier asserts

Current anchors:

- primary literature instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification instance: `constant_10_rolling_dp_reference`
- practical benchmark dataset: `retail_like_weekly_trace`
