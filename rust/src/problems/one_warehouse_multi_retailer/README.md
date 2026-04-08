# one_warehouse_multi_retailer

Rust-first problem home for `one_warehouse_multi_retailer`.

Repo interpretation:

- one upstream warehouse
- multiple downstream retailers
- coupled replenishment and allocation decisions

Code lives under `rust/src/problems/one_warehouse_multi_retailer/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface:

- `env.rs` exposes raw state quantities only
- any normalization or derived feature encoding for policies must live outside the environment layer
- `literature_verified` is the only verification-status label carried by references in this package

Benchmark notes:

- proportional rationing must exhaust available warehouse inventory; floor-only rounding is not a valid benchmark implementation
- the current Kaynov Table A.3 reproduction uses a mean-filled pipeline warm start in the script layer rather than an empty-system cold start
- learned-policy benchmarks may train with `random_sequential` allocation and evaluate with `proportional`, matching the policy-training protocol discussed by Kaynov et al. (2024)
- for symmetric retailer instances, the preferred learned action space is `symmetric_echelon_targets`: one warehouse target and one shared retailer target, expanded into retailer orders inside the rollout layer
