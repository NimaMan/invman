# Problems

`src/problems/` is the canonical home for Rust-first problem families.

Real-world source-backed applications built on top of those families live separately under
`src/case_studies/`.

Each mature Rust-first family should keep its code and artifacts co-located under
`src/problems/<problem>/`, including:

- executable code
- literature notes
- practical benchmark assets
- experiment definitions
- human-readable verification targets

The old Python-first `invman/problems/` packages have been removed. Python now acts as a tier-two
support layer through the flattened modules in `invman/` plus benchmark glue under `scripts/`;
problem dynamics, reference instances, heuristics, exact solvers, and rollout kernels live here in
`src/problems/`.

Current Rust-first families:

- `perishable_inventory`
- `nonstationary_lot_sizing`
- `random_yield_inventory`
- `joint_replenishment`
- `one_warehouse_multi_retailer`
- `decentralized_inventory_control`
- `production_assembly_distribution_network`
- `spare_parts_inventory`
- `ameliorating_inventory`
- `procurement_removal_inventory`
- `vendor_managed_inventory`
- `joint_pricing_inventory`

Learned policy descriptors stay in `invman/policy.py`, with policy-name parsing in
`invman/policy_registry.py` and Rust-backed fitness dispatch in `invman/rollout_fitness.py`.
Problem folders own the environment, baseline heuristics, rollout path, and reference benchmarks.

Markdown convention:

- each folder uses a single markdown entrypoint
- that file is always `README.md`

Standard layout:

```text
src/problems/<problem>/
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

- the canonical first implementation lives under `src/problems/<problem>/`
- Python can mirror the structure later if we need a higher-level package wrapper
- the Rust module must already contain the environment, heuristic baselines, rollout path, and
  verification anchors before the family counts as implemented

Rust also has a descriptive cross-problem layer under `src/problems/core/`.

And the crate now has a separate case-study layer under `src/case_studies/` for concrete
systems such as Hormuz. Those folders are expected to use the same FlowNet language, but they are
not treated as reusable benchmark families.

That layer is not another simulator. It is the shared problem blueprint for the repo. The canonical
entrypoints for that design are:

- `src/problems/core/README.md`
- `src/problems/core/flownet/`

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
- `multi_echelon/serial`
  - the textbook serial multi-echelon system (Clark & Scarf 1960). Its `env.rs` is
    literature-verified BY SIMULATION: driven by the optimal echelon base-stock policy it
    reproduces Snyder & Shen Example 6.1 optimal cost 47.65 and the discrete Poisson optima
    (3-stage 72.04, etc.) within Monte-Carlo error
  - the `exact` solver reproduces the same optima analytically (within 0.05%), cross-checked
    against Snyder's public `stockpyl.ssm_serial`; exact and simulation agree
  - this is the clean, training-ready env for the serial problem
- `multi_echelon/assembly` (NOT literature-verified — verified BY EQUIVALENCE only)
  - the textbook assembly system (Rosling 1989): components procured from outside suppliers are
    assembled into one finished product. By Rosling (1989) it is equivalent to a serial system; the
    equal-lead-time reduction in `rosling.rs` collapses it to a 2-stage `kit → finished` serial
    system, and the env-sim reproduces that serial optimum (finished lead time 1)
  - it carries NO directly reproducible PUBLISHED assembly number, so every carried instance is
    `literature_verified = false` (`assembly/references.rs`, guarded by
    `references::tests::no_assembly_instance_is_literature_verified`). Rosling (1989) is a
    structural result (no worked assembly cost/base-stock table), and the only published number in
    the chain (Snyder & Shen Example 6.1 = 47.65) is a 3-stage serial system the 2-stage assembly
    reduction cannot reach
  - honest basis: literature-verified at the STRUCTURAL/equivalence level (Rosling 1989) + the env
    reproduces, by simulation, the optimum of the literature-verified serial solver it reduces to.
    The assembly instance numbers (22.759 / 52.536 / 27.530) are solver-derived, not published
- `production_assembly_distribution_network` (NOT literature-verified)
  - this family is the richer Pirhooshyaran & Snyder (2021) general supply-network model, NOT the
    textbook serial system. Its `env.rs` adds per-node production steps and pipeline holding, so
    it does not reproduce the textbook serial optimum (the `serial_echelon_simulation.rs` test
    shows the structural gap quantitatively: ~147 / >100 vs 72.04)
  - the single-node newsvendor rows are reproduced analytically; the serial benchmark rows it
    carries are the textbook Clark-Scarf optima, verified in the `multi_echelon/serial` family
  - the Pirhooshyaran env's own published serial protocol could not be recovered from public
    sources, so this env stays not literature-verified

Everything else should be treated as not literature-verified unless the problem README states
otherwise explicitly.

Notable current example:

- `ameliorating_inventory`
  - repo-exact verified on its reduced finite-horizon verifier
  - not literature-verified, because the executable package is a reduced approximation of the
    Pahr/Grunow ameliorating-food model and the reported benchmark performance belongs to the
    richer stochastic price/decay and LP-blending setup
- `joint_replenishment`
  - repo-exact verified on its reduced two-item finite-horizon verifier
  - not literature-verified, because the carried Vanvuchelen et al. settings expose public
    instance definitions and relative figures but not exact per-setting benchmark rows suitable for
    repo assertions
- `procurement_removal_inventory`
  - repo-exact verified on its reduced finite-horizon verifier
  - not literature-verified, because the executable package is a simplified procurement/removal
    inventory-control slice while the Maggiar/Sadighian anchor is a richer pricing and revenue
    management model without exact public rows for this repo package
- `random_yield_inventory`
  - repo-exact verified on its reduced finite-horizon all-or-nothing verifier
  - not literature-verified, because Yan et al. matches the model family but does not expose public
    row-level benchmark numbers, while Inderfurth/Kiesmuller reports numbers for related broader
    random-yield models rather than this repo executable
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

- `src/problems/<problem>/experiments/README.md`

Typical generated outputs:

- `src/problems/<problem>/experiments/reports/latest_report.json`
- `src/problems/<problem>/experiments/reports/README.md`
