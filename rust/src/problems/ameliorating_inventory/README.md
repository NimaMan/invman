# ameliorating_inventory

Rust-first problem home for `ameliorating_inventory`.

Problem formulation in the source paper:

- age-structured inventory with ameliorating products
- multiple saleable products with different target ages
- stochastic demand, stochastic sales prices, and stochastic decay
- purchase decisions coupled with a blending-based issuance subproblem

Current Rust interpretation:

- discrete reduced benchmark model
- fixed product prices and fixed age-retention profile
- purchase control plus exact average-age issuance search
- repo-native exact verifier on a reduced instance

Current status:

- not literature-verified
- repo-exact verified on the reduced verifier

Reason:

- the paper and companion repository use a materially richer executable model than the current Rust
  package
- the public companion defaults use ten age classes, three products, stochastic sales prices, and
  stochastic beta decay processes
- the current Rust package is a tractable approximation of that family, not a faithful executable
  port

Package layout:

- literature references: `literature/references.rs`
- verification code: `verification/tests.rs`
- exact reduced solver: `finite_horizon_dp.rs`
- heuristics: `heuristics/`
- rollout path: `rollout.rs`
- practical notes: `practical/`
- experiment notes: `experiments/`
