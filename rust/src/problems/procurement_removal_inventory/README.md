# procurement_removal_inventory

Rust-first problem home for `procurement_removal_inventory`.

Repo interpretation:

- single-item procurement plus removal control
- returnable quota state
- joint purchase and removal action each period
- simplified inventory-control slice only; the full Maggiar/Sadighian paper model also includes
  pricing, selling revenue, markdowns, and richer demand response

Code lives under `rust/src/problems/procurement_removal_inventory/`.

Verification and benchmark anchors live in:

- `literature/references.rs`
- `verification/tests.rs`
- `practical/`
- `experiments/`

Current status:

- literature-verified: no
- repo-exact verified: yes on the reduced finite-horizon verifier
- the Maggiar/Sadighian papers are carried as structural anchors, but the current Rust package is
  not the full pricing/revenue-management model and no exact public row verifies this simplified
  package

State interface:

- `env.rs` exposes raw state quantities only
- the soft-tree benchmark uses an explicit policy-side feature map in `rollout.rs`
- normalization or derived ratios must not be hidden inside the environment layer
