# Problems

`rust/src/problems/` is the canonical home for Rust-first problem families.

Real-world source-backed applications built on top of those families live separately under
`rust/src/case_studies/`.

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
    README.md
    mod.rs
    references.rs
  practical/
    README.md
    mod.rs
    datasets/
    reports/
  experiments/
    README.md
    mod.rs
    reports/
  verification/
    README.md
    mod.rs
    tests.rs
  mod.rs
  env.rs
  heuristics/
  rollout.rs
  bindings.rs
```

## Direction

New problem families are Rust-first.

That means:

- the canonical first implementation lives under `rust/src/problems/<problem>/`
- Python can mirror the structure later if we need a higher-level package wrapper
- the Rust module must already contain the environment, heuristic baselines, rollout path, and
  verification anchors before the family counts as implemented

Rust also has a descriptive cross-problem layer under `rust/src/problems/core/`.

And the crate now has a separate case-study layer under `rust/src/case_studies/` for concrete
systems such as Hormuz. Those folders are expected to use the same FlowNet language, but they are
not treated as reusable benchmark families.

That layer is not another simulator. It is the shared problem blueprint for the repo. The canonical
entrypoints for that design are:

- `rust/src/problems/core/README.md`
- `rust/src/problems/core/flownet/`

## State Interface Rule

Environment-state rule:

- `env.rs` owns the raw environment state and transition logic
- `env.rs` may expose raw state vectors for policies, but those vectors should be direct state
  quantities in a stable order
- `env.rs` should not silently normalize, rescale, ratio-encode, or otherwise transform state for
  a learned policy
- if a learned policy uses scaled or derived features, that conversion belongs in the policy or
  rollout layer and must be explicit there
- tests should freeze the raw state ordering separately from any policy-specific feature encoding

## Reference Rule

Reference and verification rule:

- literature-backed repo assertions require publicly reported benchmark numbers from the paper
- if a paper gives the model or heuristic but not usable benchmark numbers, it can still be cited
  as background, but not as a verification anchor
- in that case the problem should use a repo-native exact solver instance for implementation
  verification and label it explicitly as `not literature-verified`

Algorithm-row rule:

- `literature_verified` applies to repo exact algorithms and repo heuristic implementations
- that label means the repo implementation has at least one public literature benchmark anchor with
  matching reported numbers for that algorithm family
- `references.rs` should store literature rows and problem-instance definitions only
- repo-generated exact or heuristic outputs should not be frozen inside `references.rs`; generate
  them in Rust during verification or store them in validation artifacts instead
- published learned-policy rows from papers, such as PPO or A3C, should be carried as published
  rows, not labeled as `literature_verified` repo algorithms
- experiment reports should separate published paper numbers from repo reproduced absolute costs

This is a cross-problem reporting principle for every benchmark family in the repo, not a
problem-specific convention.

Current literature-verified package anchors:

- `lost_sales`
  - executable heuristic reproduction for the standard benchmark heuristics currently covers
    `myopic1`, `myopic2`, and `svbs`
  - not every carried literature row in that family is executable or verified
- `dual_sourcing`
  - the bounded-DP benchmark layer reproduces the published Gijsbrechts et al. (2022) Figure 9
    optimality-gap labels for the carried six-instance family
- `lost_sales_fixed_order_cost`
  - the Rust exact solver and exact heuristic evaluators reproduce the published Bijvank et al.
    (2015) Table 1 validation instance
- `spare_parts_inventory`
  - the Kranenburg Chapter 5 exact benchmark family is literature-verified
- `decentralized_inventory_control`
  - the classic Sterman / Caner Beer-Game benchmark is literature-verified through the exact public verification code path
- `vendor_managed_inventory`
  - the public Gosavi/Sui/Giannoccaro/Lin worked newsvendor calculation is literature-verified
    through the Rust analytical verification helper
  - the full Giannoccaro and Pontrandolfo (2010) 8-case truck-dispatch profit table is not carried
    as a verified benchmark because the public demand-signal semantics do not reproduce the rows

Everything else should be treated as not literature-verified unless the problem README states
otherwise explicitly.

Notable current example:

- `joint_replenishment`
  - repo-exact verified on its reduced two-item finite-horizon verifier
  - not literature-verified, because the carried Vanvuchelen et al. settings expose public
    instance definitions and relative figures but not exact per-setting benchmark rows suitable for
    repo assertions
- `joint_pricing_inventory`
  - repo-exact verified on its reduced verifier
  - not literature-verified, because the carried Zhou/Qin anchors do not currently expose a clean
    public executable benchmark row for this repo package

Every new problem family must have:

- a canonical literature interpretation
- a literature references file that is the source of truth for problem instances and benchmark
  numbers
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
- `literature/references.rs`
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
- `verification/tests.rs`
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
