# joint_replenishment

Rust-first problem home for `joint_replenishment`.

Repo interpretation:

- multi-item replenishment
- shared full-truckload replenishment cost
- item-specific demand and inventory costs
- feasible raw actions are item order quantities whose aggregate quantity is either zero or an exact
  multiple of the truck capacity

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
- `env.rs` validates raw action feasibility, including the full-truckload multiple constraint
- the current soft-tree benchmark keeps any aggregate or normalized policy features in `rollout.rs`
- learned-policy actions are converted to feasible full-truckload quantities in `rollout.rs` before
  entering the environment
- environment code must not hide learned-policy preprocessing
