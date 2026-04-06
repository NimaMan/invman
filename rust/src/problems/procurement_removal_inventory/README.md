# procurement_removal_inventory

Rust-first problem home for `procurement_removal_inventory`.

Repo interpretation:

- single-item procurement plus removal control
- returnable quota state
- joint purchase and removal action each period

Code lives under `rust/src/problems/procurement_removal_inventory/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface:

- `env.rs` exposes raw state quantities only
- the soft-tree benchmark uses an explicit policy-side feature map in `rollout.rs`
- normalization or derived ratios must not be hidden inside the environment layer
