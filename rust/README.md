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
- `src/case_studies/<case_study>/`
  - source-backed real-world applications built on top of the problem layer
  - scenario engines, reproducible datasets, and public-facing artifacts for a concrete system
- `src/problems/core/`
  - cross-problem FlowNet layer
  - the fundamental questions every inventory problem should answer
  - physical, stochastic, control, objective, and timing skeletons
- `src/problems/<problem>/`
  - canonical Rust-side home for reusable executable families and their human-readable artifacts
- `src/case_studies/<case_study>/`
  - canonical Rust-side home for concrete source-backed applications that use those families or the
    FlowNet language directly

Current problem modules:

- `src/problems/lost_sales/`
- `src/problems/lost_sales_fixed_order_cost/`
- `src/problems/dual_sourcing/`
- `src/problems/multi_echelon/`
- `src/problems/perishable_inventory/`
- `src/problems/nonstationary_lot_sizing/`
- `src/problems/random_yield_inventory/`

Current case studies:

- `src/case_studies/hormuz_strait/`

## Standard Module Contract

All new Rust problem families should use one canonical folder under `src/problems/`.

Required files:

```text
src/problems/<problem>/
  mod.rs
  README.md
  literature/
  practical/
    datasets/
    reports/
  experiments/
  verification/
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

- descriptive helper modules such as `demand.rs`, `supply.rs`, `allocation.rs`, `policies.rs`,
  `finite_horizon_dp.rs`, `value_iteration_mdp.rs`, or `rolling_scarf_dp.rs` when the problem
  structure requires them

File responsibilities:

- `env.rs`: state definition, transition logic, and period cost accounting
- `heuristics/`: classical benchmark policies, split by policy family
- `rollout.rs`: learned-policy rollout kernels used by training and evaluation
- `references.rs`: literature instances, published values, repo canonical instance, and
  `VERIFICATION_PROBLEM_INSTANCE`
- `bindings.rs`: Python-facing entrypoints
- `tests/verification.rs`: the exact correctness anchor for the problem dynamics and heuristics
- `src/problems/<problem>/literature/`: the human-readable interpretation of the carried paper family
- `src/problems/<problem>/practical/datasets/`: checked-in practical benchmark traces or descriptors
- `src/problems/<problem>/practical/reports/`: checked-in canonical benchmark snapshots
- `src/problems/<problem>/experiments/`: paper-facing experiment definitions for reported benchmark
  studies
- `src/problems/<problem>/verification/`: human-readable targets for what the tests assert
- problem-specific exact or search-style helper modules should stay clearly named, such as
  `finite_horizon_dp.rs`, `value_iteration_mdp.rs`, `policy_evaluation.rs`, or
  `rolling_scarf_dp.rs`

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

## What counts as "literature-verified"

A family (or a reference instance's `literature_verified = true` flag) is **literature-verified
only when an in-crate test RE-RUNS the env** — and at least one benchmark heuristic/solver — **and
asserts the freshly computed cost / policy / action reproduces a number printed in a paper, within a
stated tolerance.** "Our run reproduces the published number" is the bar.

- A **frozen snapshot** test — one that only asserts a carried table of constants equals the same
  published constants (`assert_eq!(CARRIED_TABLE, [the_published_literals])`) — is **NOT
  verification**. It is at most a drift-guard, and must be paired with an executing reproduction
  test that actually drives the env.
- **Self-consistency** with our own DP/MDP, or **matching another code library** (e.g. `stockpyl`),
  is a reference-implementation match, **not** literature verification — record it as `partial`.
- Set `literature_verified = true` only when such an executing reproduction test passes. Otherwise
  keep it `false` and state the honest status (reference-impl / figure-only / structurally
  unreachable by this MDP / distinct MDP). It is correct and expected for some envs to stay
  `false` with an honest characterization (negative) test instead.

## Problem-Space Backbone

`src/problems/core/` is the FlowNet layer above the executable problem modules.

It does not replace `env.rs` or `rollout.rs`. Instead it defines the common modeling questions
behind any problem family:

- what inventory states exist
- how material moves or transforms
- what random events occur
- what the controller can choose
- what the controller can observe, and when
- how performance is scored
- what timing rules and constraints shape the system

Those questions are then organized into five layers:

- physical
- stochastic
- control
- objective
- timing

The detailed design notes for that backbone live in `src/problems/core/README.md`, and the
canonical problem-language types live in `src/problems/core/flownet/`.

Rules for the first test:

- every new family must ship with one verification test before policy training work starts
- the verification test should prove both:
  - environment mechanics are correct
  - at least one benchmark heuristic produces the expected result on the verification instance
- if exact tabular verification logic is needed, put it in a clearly named module such as
  `finite_horizon_dp.rs` or `value_iteration_mdp.rs` rather than embedding it directly in
  `tests/verification.rs`

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
