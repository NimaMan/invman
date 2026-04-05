# Verification Targets

Executable verifier:

- `rust/src/problems/nonstationary_lot_sizing/tests/verification.rs`

What the verifier asserts:

1. reference-set shape
   - 8 forecast definitions are present
   - 8 benchmark instances are present
   - the primary reference instance is `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
2. observation and mechanics checks
   - the policy-state layout matches the Section 4.1 formulation
   - the Section 4.2 worked transition has reward `-130`
3. simple baseline checks
   - the literature formula gives `(s, S) = (33.351246609652, 47.49338223338295)` on the primary
     constant-10 window
   - the simulated simple baseline matches the author reference row within the stored tolerances
4. rolling-DP checks
   - the first-period rolling-DP levels are `(28, 42)`
   - the rolling-DP simulated benchmark matches the author reference row within the stored
     tolerances

Verification semantics:

- the worked transition is literature-backed
- the benchmark-row assertions use the published author-repo reference CSVs and the tolerances in
  `VERIFICATION_PROBLEM_INSTANCE`
