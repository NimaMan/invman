# spare_parts_inventory

Rust-first problem home for `spare_parts_inventory`.

Repo interpretation:

- repairable spare-parts control
- installed-base failures create demand
- procurement and repair pipelines jointly determine service and downtime
- this folder may also catalog adjacent spare-parts literature benchmarks when a paper publishes
  reusable numeric benchmark rows, even if the repo-native executable primary instance is a
  different spare-parts subfamily

Current benchmark split:

- executable repo-native primary instance: single-echelon repairable spare-parts control
- executable literature-verified exact benchmark: Kranenburg (2006) Chapter 5 lateral-transshipment comparison
- literature-verified catalog: van Oers et al. (2024) two-echelon periodic-review serial
  spare-parts benchmark with optional additive manufacturing

Code lives under `rust/src/problems/spare_parts_inventory/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface rule:

- `env.rs` exposes raw state quantities only
- any normalization, scaling, or derived inventory-position features for learned policies must live outside the environment layer
- `rollout.rs` is the right place to convert raw state into the feature vector expected by a specific policy family
