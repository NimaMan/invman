# joint_replenishment

Rust-first problem home for `joint_replenishment`.

Repo interpretation:

- multi-item replenishment
- shared major ordering or truck cost
- item-specific demand and inventory costs

Code lives under `rust/src/problems/joint_replenishment/`.

Verification and benchmark anchors live in:

- `literature/references.rs`
- `verification/tests.rs`
- `practical/`
- `experiments/`

Current status:

- literature-verified: no
- repo-exact verified: yes on the reduced two-item finite-horizon verifier
- the 16 Vanvuchelen small-scale settings are carried as public problem definitions, but the paper
  does not provide exact per-setting benchmark rows suitable for repo assertions

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps any aggregate or normalized policy features in `rollout.rs`
- environment code must not hide learned-policy preprocessing
