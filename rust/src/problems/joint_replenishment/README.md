# joint_replenishment

Rust-first problem home for `joint_replenishment`.

Repo interpretation:

- multi-item replenishment
- shared major ordering or truck cost
- item-specific demand and inventory costs

Code lives under `rust/src/problems/joint_replenishment/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps any aggregate or normalized policy features in `rollout.rs`
- environment code must not hide learned-policy preprocessing
