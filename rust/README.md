# invman_rust

Rust extension module for high-throughput inventory-management rollouts.

Current scope:

- native rollout kernels for:
  - lost sales
  - fixed-order-cost heuristic search
  - dual sourcing
  - multi-echelon
  - perishable inventory
- batched soft-tree population evaluation for CMA-ES
- soft-tree policy support for:
  - `oblique` and `axis_aligned` split types
  - `constant` and `linear` leaf outputs

## Source layout

The crate mirrors the Python package structure:

- `src/core/`
  - shared Rust-native runtime pieces
  - `policies/` holds reusable backbone math such as dense networks and soft trees
- `src/problems/<problem>/`
  - problem-local environment transitions
  - rollout kernels
  - heuristic search
  - problem-specific action mappings when needed

Current problem modules:

- `src/problems/lost_sales/`
- `src/problems/lost_sales_fixed_order_cost/`
- `src/problems/dual_sourcing/`
- `src/problems/multi_echelon/`
- `src/problems/perishable_inventory/`
- `src/problems/nonstationary_lot_sizing/`
- `src/problems/random_yield_inventory/`

## Standard Module Contract

All new Rust problem families should use the same folder contract.

Required files:

```text
src/problems/<problem>/
  mod.rs
  env.rs
  heuristics/
    mod.rs
  rollout.rs
  references.rs
  bindings.rs
  tests/
    mod.rs
    verification.rs
```

Optional files:

- `exact.rs`, `demand.rs`, `supply.rs`, `allocation.rs`, `policies.rs`, `dp.rs`, or similar
  helpers when the problem structure requires them

File responsibilities:

- `env.rs`: state definition, transition logic, and period cost accounting
- `heuristics/`: classical benchmark policies and heuristic search helpers, split by policy family
- `rollout.rs`: learned-policy rollout kernels used by training and evaluation
- `references.rs`: literature instances, published values, repo canonical instance, and
  `VERIFICATION_PROBLEM_INSTANCE`
- `bindings.rs`: Python-facing entrypoints
- `tests/verification.rs`: the exact correctness anchor for the problem dynamics and heuristics
- `exact.rs` when needed: exact tabular solvers or analytical verification helpers used to
  reproduce literature anchors cleanly outside the test file

Rules for `references.rs`:

- it is the source of truth for the literature instances we keep in the repo
- it must contain every paper instance we want to benchmark later, not only the first one we test
- it must define:
  - `PRIMARY_REFERENCE_INSTANCE`
  - `VERIFICATION_PROBLEM_INSTANCE`
- it must distinguish:
  - exact published values
  - repo-native benchmark values
  - deterministic worked-example values used only for correctness testing

Rules for the first test:

- every new family must ship with one verification test before policy training work starts
- the verification test should prove both:
  - environment mechanics are correct
  - at least one benchmark heuristic produces the expected result on the verification instance
- if exact tabular verification logic is needed, put it in `exact.rs` rather than embedding it
  directly in `tests/verification.rs`

Build into the active project virtualenv with:

```bash
python scripts/build_rust_extension.py
```

## Current best native-backed tree result

On the trusted vanilla lost-sales benchmark, the best tree architecture using this native path is:

- oblique split structure
- depth `2`
- linear leaf outputs
- mean cost `4.753725`

This currently outperforms the heuristic baseline `Myopic-2 = 4.8204`.
