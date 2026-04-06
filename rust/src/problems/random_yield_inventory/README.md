# random_yield_inventory

Rust-first problem home for `random_yield_inventory`.

Repo interpretation:

- single-item inventory with stochastic supply yield
- positive lead time
- heuristic and exact reduced benchmark support on discrete instances

Code lives under `rust/src/problems/random_yield_inventory/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps its derived feature map in `rollout.rs`
- any normalization or expectation-based encoding must stay outside the environment layer
