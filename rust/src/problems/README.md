# Problems

`rust/src/problems/` is the canonical home for Rust-first problem families.

Each mature Rust-first family should keep its code and artifacts co-located under
`rust/src/problems/<problem>/`, including:

- executable code
- literature notes
- practical benchmark assets
- experiment definitions
- human-readable verification targets

Legacy Python-first families still live under `invman/problems/`:

- `lost_sales`
- `lost_sales_fixed_order_cost`
- `dual_sourcing`
- `multi_echelon`

Those packages are still active runtime dependencies for scripts, tests, and policy dispatch. They
should not be removed until they are fully migrated.

Current Rust-first families:

- `perishable_inventory`
- `nonstationary_lot_sizing`
- `random_yield_inventory`
- `joint_replenishment`
- `one_warehouse_multi_retailer`
- `decentralized_inventory_control`
- `network_inventory`
- `spare_parts_inventory`
- `ameliorating_inventory`
- `procurement_removal_inventory`
- `vendor_managed_inventory`
- `joint_pricing_inventory`

Learned policy classes stay separate under `invman/policies/`. Problem folders own the environment,
baseline heuristics, rollout path, and reference benchmarks.

Markdown convention:

- each folder uses a single markdown entrypoint
- that file is always `README.md`

Standard layout:

```text
rust/src/problems/<problem>/
  README.md
  literature/
  practical/
    datasets/
    reports/
  experiments/
    reports/
  verification/
  mod.rs
  env.rs
  heuristics/
  rollout.rs
  references.rs
  bindings.rs
  tests/
```

## Direction

New problem families are Rust-first.

That means:

- the canonical first implementation lives under `rust/src/problems/<problem>/`
- Python can mirror the structure later if we need a higher-level package wrapper
- the Rust module must already contain the environment, heuristic baselines, rollout path, and
  verification anchors before the family counts as implemented

Rust also has a descriptive cross-problem layer under `rust/src/problems/core/`.

That layer is not another simulator. It is the shared problem blueprint for the repo. The canonical
entrypoints for that design are:

- `rust/src/problems/core/README.md`
- `rust/src/problems/core/flownet/`

## Reference Rule

Reference and verification rule:

- literature-backed repo assertions require publicly reported benchmark numbers from the paper
- if a paper gives the model or heuristic but not usable benchmark numbers, it can still be cited
  as background, but not as a verification anchor
- in that case the problem should use a repo-native exact solver instance for implementation
  verification and label it explicitly as `not literature-verified`

Every new problem family must have:

- a canonical literature interpretation
- a references file that is the source of truth for problem instances and benchmark numbers
- at least one heuristic baseline
- rollout code for learned-policy training and evaluation
- one verification instance with passing assertions before any training work starts

What `implemented` means in this repo:

- the environment dynamics are coded
- the baseline heuristics are coded
- the rollout path used later for learned policies is coded
- the literature instances we care about are recorded
- at least one verification test runs our implementation and asserts expected numbers

## Required Artifacts

Required artifacts for every new Rust-first problem family:

- `README.md`
  - short literature note
  - benchmark scope
  - canonical repo interpretation of the family
- `references.rs`
  - authoritative list of literature instances carried by the repo
  - one `PRIMARY_REFERENCE_INSTANCE`
  - one `VERIFICATION_PROBLEM_INSTANCE`
  - published numbers when they exist
  - explicit notes when repo values are repo-native rather than verbatim literature values
- `env.rs`
  - state transition logic
  - state validation
  - period cost accounting
- `heuristics/`
  - classical benchmark policies for that family
- `rollout.rs`
  - learned-policy evaluation path
  - deterministic rollout helpers from fixed paths when needed
- `tests/verification.rs`
  - assertions tied to the verification instance

If verification needs an exact finite-state solver or analytical evaluator, keep it outside
`heuristics/` in a clearly named role-specific module such as `finite_horizon_dp.rs`,
`value_iteration_mdp.rs`, `policy_evaluation.rs`, or `rolling_scarf_dp.rs`.

## Benchmark Layers

Every mature family should eventually support three benchmark layers:

1. `verification`
   - tiny exact or frozen reference instance
   - purpose: prove implementation correctness
   - expected output: assertions in tests
2. `literature`
   - benchmark settings and policy rows carried from papers
   - purpose: show that the repo reproduces the published family and baseline behavior
   - expected output: reference metadata plus validation scripts where possible
3. `practical`
   - trace-backed or dataset-backed evaluation closer to operations use
   - purpose: evaluate policies outside the narrow verification setting
   - expected output: benchmark scripts, dataset descriptors, and markdown/json reports

We do not skip verification in favor of practical benchmarks.

## Experiments

Each mature Rust-first family should also carry an `experiments/` folder.

This is the paper-reporting layer. It defines:

- which instances we report
- which learned policy families we optimize with CMA-ES
- which heuristic baselines we compare against
- whether an exact or near-exact benchmark exists
- which metrics we report in paper tables

The default paper-facing file for a mature family is:

- `rust/src/problems/<problem>/experiments/README.md`

Typical generated outputs:

- `rust/src/problems/<problem>/experiments/reports/latest_report.json`
- `rust/src/problems/<problem>/experiments/reports/README.md`
