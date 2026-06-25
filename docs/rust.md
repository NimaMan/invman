# Rust Crate

`invman_rust` is the Rust-first execution layer for `invman`.

Rust owns:

- environment transition logic
- exact, bounded, or reduced dynamic programs
- heuristic evaluators and search routines
- learned-policy rollout kernels
- high-throughput population rollouts used by CMA-ES
- PyO3 bindings exposed through the `invman_rust` Python module

Python owns orchestration: policy descriptors, optimization loops, experiment
scripts, report generation, and calls into the Rust bindings. Python should not
duplicate environment dynamics or learned-policy inference when a Rust rollout
exists.

## Build Model

The crate is rooted at `Cargo.toml` and `src/lib.rs`.

Important Cargo facts:

- crate name: `invman_rust`
- Rust edition: `2021`
- library outputs: `rlib` and `cdylib`
- default features: empty, for Rust-native development and tests
- `python-extension`: enables PyO3 `extension-module` for maturin builds
- `ppo`: optional Candle-backed reusable Rust PPO trainer

Use the repo helper to build the Python extension into the active environment:

```bash
python scripts/build_rust_extension.py
```

Run Rust-native tests from the repo root:

```bash
cargo test --manifest-path Cargo.toml -q
```

After changing `bindings.rs` or `src/lib.rs`, rebuild the extension before
running Python tests that import `invman_rust`.

## Source Layout

```text
src/lib.rs                 PyO3 module registration
src/core/                  shared Rust policy and PPO infrastructure
src/problems/              reusable benchmark problem families
src/problems/core/         FlowNet descriptive problem-language layer
src/case_studies/          source-backed applications built on the problem layer
```

`src/core/policies/` contains reusable policy math such as dense policies and
soft trees. `src/core/ppo/` contains the optional reusable PPO trainer.

`src/problems/core/` is not another simulator. It is the cross-problem FlowNet
language for physical, stochastic, control, objective, and timing structure. See
`src/problems/core/README.md` and `src/problems/core/flownet/`.

`src/case_studies/` is for concrete source-backed systems, such as
`hormuz_strait`, not reusable benchmark families.

## Registered Python Bindings

`src/lib.rs` currently registers:

- core policy helpers: `core::policies::bindings`
- `ameliorating_inventory`
- `decentralized_inventory_control`
- `lost_sales::vanilla`
- `lost_sales::fixed_order_cost`
- `dual_sourcing`
- `joint_replenishment`
- `joint_pricing_inventory`
- `multi_echelon`
- `nonstationary_lot_sizing`
- `one_warehouse_multi_retailer`
- `perishable_inventory`
- `procurement_removal_inventory`
- `random_yield_inventory`
- `spare_parts_inventory`
- `vendor_managed_inventory`
- case study: `hormuz_strait`

If a Rust function should be callable from Python, add the `#[pyfunction]` in
the problem's `bindings.rs`, register it in that module's `register_py`, then
ensure the module is registered in `src/lib.rs`.

## Problem Families

Reusable executable families live under `src/problems/`.

Current top-level families:

- `ameliorating_inventory`
- `decentralized_inventory_control`
- `dual_sourcing`
- `joint_pricing_inventory`
- `joint_replenishment`
- `lost_sales`
- `multi_echelon`
- `nonstationary_lot_sizing`
- `one_warehouse_multi_retailer`
- `perishable_inventory`
- `procurement_removal_inventory`
- `random_yield_inventory`
- `spare_parts_inventory`
- `vendor_managed_inventory`

Important nested families:

- `lost_sales/vanilla`
- `lost_sales/fixed_order_cost`
- `multi_echelon/serial`
- `multi_echelon/assembly`
- `multi_echelon/divergent_special_delivery`
- `multi_echelon/general_backorder_fixed_cost`
- `multi_echelon/production_assembly_distribution_network`

Start with `src/problems/README.md` and the target problem's `README.md` before
editing a family. Those files carry the current verification status and caveats.

## Standard Problem Contract

New mature Rust-first problem families should converge on this shape:

```text
src/problems/<problem>/
  README.md
  mod.rs
  env.rs
  rollout.rs
  bindings.rs
  instances/
    README.md
    <instance_id>.json
  literature/
    README.md
    mod.rs
    references.rs
  practical/
    README.md
    datasets/
    reports/
  experiments/
    README.md
  verification/
    README.md
    mod.rs
    tests.rs
  heuristics/
    mod.rs
  tests/
    mod.rs
    verification.rs
```

Not every existing family has every folder, and umbrella families can put
instance catalogs in subproblem folders such as
`src/problems/multi_echelon/serial/instances/`.

File responsibilities:

- `env.rs`: state, transition, event timing, and cost/profit accounting
- `rollout.rs`: learned-policy rollout kernels and population rollouts
- `heuristics/`: classical policies and benchmark comparators
- `bindings.rs`: Python-facing PyO3 entrypoints
- `instances/`: machine-readable benchmark instances and provenance
- `literature/`: paper interpretation and source-backed reference definitions
- `verification/`: human-readable verification targets and executable tests
- `experiments/`: paper-facing experiment definitions and report snapshots
- `practical/`: practical traces, datasets, and practical benchmark reports

Problem-specific solver modules should be named directly, for example
`finite_horizon_dp.rs`, `value_iteration_mdp.rs`, `exact.rs`,
`policy_evaluation.rs`, or `rolling_dp.rs`.

## Instance Catalogs

The active machine-readable instance convention is:

```text
src/problems/<problem-or-subproblem>/instances/
  README.md
  <instance_id>.json
```

Use `scripts/instances/validate_problem_instances.py` to check the cross-family
schema:

```bash
python scripts/instances/validate_problem_instances.py
```

Do not add new `BENCHMARK.md` files. Keep instance provenance in
`instances/README.md` and JSON files. Keep broader interpretation in the problem
`README.md`, `literature/README.md`, and `verification/README.md`.

Instance classification values are:

- `strict_literature`
- `companion_code`
- `table_only`
- `faithful_unverified`
- `generated`

Keep repo-generated numbers separate from published numbers. A value computed by
this crate is a reproduction or verifier output, not a literature row.

## Verification Rule

A family or instance is literature-verified only when executable code reruns the
environment or solver and reproduces a public number, action, policy, or table
entry from the cited source within a stated tolerance.

This does not count as strict literature verification:

- freezing a table of constants and checking that it still equals itself
- matching a repo-native exact solver with no public number
- matching an adjacent library such as `stockpyl` without a public paper row
- carrying a published DRL/PPO/A3C row that the repo does not implement
- reproducing a related but structurally different model

Those can still be useful, but label them honestly as `companion_code`,
`table_only`, `faithful_unverified`, `generated`, or equivalent problem-local
status.

Every new problem family should have at least one executable verification test
before learned-policy training starts. The test should cover both mechanics and
at least one solver or heuristic comparator where possible.

Good verification entry points:

```bash
cargo test --manifest-path Cargo.toml -q
python scripts/instances/validate_problem_instances.py
python -m pytest tests/test_problem_verification_files.py -q
```

Use narrower cargo filters for slow families, for example:

```bash
cargo test -q serial_rows_reproduced_by_exact_clark_scarf_solver
cargo test general_backorder_fixed_cost --lib
cargo test -p invman_rust --lib problems::ameliorating_inventory::tests::verification -- --nocapture
```

## Policy Rollout Boundary

The active policy boundary is:

- Python builds a `Policy` descriptor and flat parameter vector.
- Rust validates the parameter layout, decodes actions, and runs the rollout.
- Python CMA-ES calls Rust single-candidate or population rollouts.

Shared Rust policy code lives under `src/core/policies/`. Python-side descriptor
code lives in:

- `invman/policy.py`
- `invman/policy_registry.py`
- `invman/policy_build.py`
- `invman/rollout_fitness.py`

Do not add a Python forward pass for a policy if the target problem has a Rust
rollout. Add or extend the Rust rollout/binding instead.

## Editing Checklist

When editing Rust problem code:

1. Read the problem `README.md` and `verification/README.md`.
2. Check whether the target instance lives in `instances/`.
3. Keep raw environment state in `env.rs`; put policy feature transforms in the
   rollout or policy layer.
4. Generate verifier outputs at test time instead of freezing repo-computed
   costs as literature references.
5. Rebuild `invman_rust` after binding changes.
6. Run a targeted cargo test and any Python smoke test that imports the changed
   binding.

When adding a new family, make the Rust implementation usable first:

- environment dynamics
- baseline heuristic or exact comparator
- rollout path
- binding surface, if Python needs it
- one verification test with stated tolerance
- one `instances/README.md` plus at least one JSON instance, when benchmark
  instances are part of the family
