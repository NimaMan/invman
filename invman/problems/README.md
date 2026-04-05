# Problems

Each inventory-control problem lives in its own subpackage under `invman/problems` or, for newer
families, under `rust/src/problems/` first.

For Rust-first families, the canonical problem home is split into:

- `rust/src/problems/<problem>/` for executable code only
- `rust/problems/<problem>/` for literature, practical, experiments, and verification artifacts

Markdown convention:

- each folder should expose at most one markdown entrypoint
- that file should always be named `README.md`

## Current Problem Set

Current problem families in the repo:

- `lost_sales`
- `lost_sales_fixed_order_cost`
- `dual_sourcing`
- `multi_echelon`
- `perishable_inventory`
- `nonstationary_lot_sizing`
- `random_yield_inventory`
- `joint_replenishment`
- `one_warehouse_multi_retailer`
- `decentralized_inventory_control`
- `network_inventory`
- `spare_parts_inventory`
- `ameliorating_inventory`

The first four have Python problem packages today. The later nine are currently Rust-first.

### Current Python Packages

- `lost_sales/`
  - `env.py`: simulator and rollout helpers
  - `heuristics.py`: classic baseline policies
  - `reference_instances.py`: trusted benchmark instances
  - `problem_info.py`: literature reference tables
  - `benchmark.py`: default heuristic evaluation helpers
  - `README.md`: canonical vanilla benchmark and refresh notes
- `lost_sales_fixed_order_cost/`
  - `env.py`: fixed-cost problem entrypoint built on the lost-sales simulator
  - `heuristics.py`: `(s,S)`, `(s,nQ)`, modified `(s,S,q)` search and evaluation
  - `reference_instances.py`: literature-derived benchmark grid
  - `benchmark.py`: benchmark runners for the fixed-cost grid
- `dual_sourcing/`
  - `env.py`: reduced-state dual-sourcing simulator with a 2D order action `(q_regular, q_expedited)`
  - `heuristics.py`: single-index, dual-index, capped dual-index, and tailored base-surge policies
  - `dp.py`: bounded dynamic-programming solver for the small-scale literature settings
  - `reference_instances.py`: six Gijsbrechts / Veeraraghavan-Scheller-Wolf benchmark settings,
    benchmark policy families, and published literature claims
  - `benchmark.py`: heuristic and DP benchmark helpers
- `multi_echelon/`
  - `env.py`: Van Roy / Gijsbrechts one-warehouse, many-retailer simulator
  - `heuristics.py`: constant base-stock benchmark search and evaluation
  - `reference_instances.py`: the two literature settings, benchmark policy families, and published
    literature claims
  - `benchmark.py`: default benchmark runner

### Current Rust-First Families

These families are implemented first under `rust/src/problems/<problem>/`:

- `perishable_inventory/`
  - perishable lost-sales dynamics with age-profile state
  - benchmark heuristics and tabular value-iteration verification
- `nonstationary_lot_sizing/`
  - forecast-driven single-item lot sizing with rolling-DP baseline verification
- `random_yield_inventory/`
  - all-or-nothing random-yield inventory with heuristic and exact finite-horizon verification
- `joint_replenishment/`
  - multi-item shared-major-cost replenishment with benchmark heuristics and exact reduced DP
- `one_warehouse_multi_retailer/`
  - centralized divergent distribution with allocation heuristics and exact reduced verification
- `decentralized_inventory_control/`
  - Beer-Game-style local-information serial chain with `base_stock` and `sterman_anchor_adjust`
    benchmarks plus exact reduced verification
- `network_inventory/`
  - generalized directed inventory networks with exact reduced verification on a small diamond graph
- `spare_parts_inventory/`
  - repairable spare-parts control with installed-base failures, repair returns, procurement, and
    exact reduced finite-horizon verification
- `ameliorating_inventory/`
  - age-improving inventory with purchase control, issuance subproblem, and exact reduced
    finite-horizon verification

Learned policy classes stay separate under `invman/policies/`. The problem packages own the
simulation, baseline heuristics, and reference benchmarks.

For mature Rust-first families, the non-code artifact home should live under
`rust/problems/<problem>/`.

## Current Direction

New problem families are Rust-first.

That means:

- the canonical first implementation lives under `rust/src/problems/<problem>/`
- Python can mirror the structure later if we need a higher-level package wrapper
- the Rust module must already contain the environment, heuristic baselines, rollout path, and
  verification anchors before the family counts as implemented

The newer Rust-first families currently include the nine families listed above.

## Standard For New Problems

Older Python packages were added incrementally and are not fully uniform. New problem families
should follow one standard from the start.

Every new problem family must have:

- a canonical literature interpretation
- a references file that is the source of truth for problem instances and benchmark numbers
- at least one heuristic baseline
- rollout code for learned-policy training and evaluation
- one verification instance with passing assertions before any training work starts

What “implemented” means in this repo:

- the environment dynamics are coded
- the baseline heuristics are coded
- the rollout path used later for learned policies is coded
- the literature instances we care about are recorded
- at least one verification test runs our implementation and asserts expected numbers

## Required Artifacts

Required artifacts for every new problem family:

- `README.md` when the family has a Python package
  - short literature note
  - benchmark scope
  - canonical repo interpretation of the family
- `references.rs` or `reference_instances.py`
  - authoritative list of literature instances carried by the repo
  - one `PRIMARY_REFERENCE_INSTANCE`
  - one `VERIFICATION_PROBLEM_INSTANCE`
  - published numbers when they exist
  - explicit notes when repo values are repo-native rather than verbatim literature values
- `env.rs` or `env.py`
  - state transition logic
  - state validation
  - period cost accounting
- `heuristics/` or `heuristics.py`
  - classical benchmark policies for that family
  - any parameter-search helpers required by those heuristics
- `rollout.rs` or `benchmark.py` / rollout helpers
  - learned-policy evaluation path
  - deterministic rollout helpers from fixed demand paths when needed
- tests tied to the verification instance

Required semantics:

- the references file is the authoritative list of benchmark instances used in the repo
- the primary reference instance is the canonical first benchmark for learned policies
- the verification instance is the minimal correctness anchor for the problem implementation
- verification must cover both:
  - environment mechanics
  - at least one benchmark heuristic or exact policy on that instance

## Benchmark Layers

Every mature family should eventually support three benchmark layers.

1. `verification`
   - tiny exact or frozen reference instance
   - purpose: prove implementation correctness
   - expected output: assertions in tests
2. `literature`
   - the benchmark settings and policy rows carried from papers
   - purpose: show that the repo reproduces the published problem family and baseline behavior
   - expected output: reference metadata plus validation scripts where possible
3. `practical`
   - trace-backed or dataset-backed evaluation that is closer to how inventory is used
   - purpose: evaluate policies on cost, service, and operational robustness outside the narrow
     verification setting
   - expected output: benchmark scripts, checked-in dataset descriptors, and markdown/json reports

We do not skip verification in favor of practical benchmarks. Practicalization comes after the
problem family is already correct.

## Experiment Design Contract

Each mature Rust-first family should also carry an `experiments/` folder under
`rust/problems/<problem>/`.

This is the paper-reporting layer. It is where we define:

- which instances we will report in the paper
- which learned policy families we will optimize with CMA-ES
- which heuristic baselines we will compare against
- whether an exact or near-exact optimal benchmark exists
- which metrics we will report in the paper tables

Current paper objective:

- design policy classes for `invman` problems
- optimize their parameters with CMA-ES
- compare against:
  - the problem heuristics
  - the optimal policy when an exact optimum exists
  - the strongest benchmark baseline when an exact optimum is not tractable

Minimum expected file:

- `rust/problems/<problem>/experiments/README.md`

That file should define at least:

- reported instances
- learned policy families
- heuristic comparators
- exact / optimal comparator availability
- reported metrics

## Verification Standard

Verification in this repo means: run our code and assert its outputs against frozen reference
numbers.

Accepted verification targets:

- exact published literature numbers
- exact literature-derived policy tables
- exact repo-native numbers from a small exact solver on a literature-shaped instance
- deterministic worked-example transitions when the purpose is to verify mechanics

What is not acceptable:

- “toy” numbers with no documented connection to either the literature instance or a repo-native
  exact derivation
- loose claims that a problem is verified without an actual assertion target

When a paper does not expose exact per-instance benchmark rows, the correct fallback is:

- keep the paper’s instance family in `references`
- add a small verification instance in the same family
- solve it exactly with a clearly named helper such as `finite_horizon_dp.rs`,
  `value_iteration_mdp.rs`, or `rolling_scarf_dp.rs`
- freeze those repo-native exact outputs in `VERIFICATION_PROBLEM_INSTANCE`
- label them explicitly as repo-native, not literature-quoted

## Practical Benchmark Contract

“Consolidate and practicalize” means pushing the current family set toward a common operational
benchmarking standard rather than only adding more problem families.

For a practical benchmark, define:

- one checked-in dataset descriptor under `rust/problems/<problem>/practical/datasets/`
- one runner under `scripts/<problem>/run_practical_benchmark.py`
- one standard report with:
  - dataset metadata
  - calibration protocol, if any
  - policy rows
  - operational metrics
- one checked-in markdown/json report under `rust/problems/<problem>/practical/reports/`

The standard report should include, when meaningful for the family:

- mean period cost or total cost
- fill rate or shortage rate
- cycle service level
- waste rate or backlog / downtime proxy
- mean holding inventory
- mean order quantity
- positive-order frequency

Not every practical benchmark needs a train/test split. Use the split only when a heuristic needs
parameter calibration. Forecast-adaptive policies can be evaluated directly on a rolling forecast
path.

Accepted practical data sources:

- real public data
- semi-real traces derived from public operational patterns
- repo-curated trace sets designed to exercise realistic decision tradeoffs

For any practical benchmark, be explicit about which of those three categories it belongs to.

## First Practicalized Families

The first two families we are practicalizing are:

- `nonstationary_lot_sizing`
  - practical reason: forecast-driven decision making is already close to real replenishment
  - current practical artifact: checked-in rolling forecast trace plus cost/service benchmark script
- `perishable_inventory`
  - practical reason: waste-service tradeoffs are central and easy to expose with trace-backed
    demand blocks
  - current practical artifact: checked-in grocery-like demand trace plus train/test heuristic
    benchmark script

These practical benchmarks do not replace the literature references in `references.rs`; they sit on
top of them.

## Reference Semantics

`references.rs` / `reference_instances.py` must distinguish clearly between:

- literature instances
  - problem settings and benchmark rows we want to carry forward from papers
- primary reference instance
  - the first canonical benchmark for learned policies in this repo
- verification problem instance
  - the smallest correctness anchor that we actually assert in tests
- worked transition references
  - single-step accounting checks for the environment
- repo-native exact values
  - exact numbers produced by our verifier for a reduced or bounded instance

If a number is not directly printed in the paper, do not present it as a literature number.

## Env/Policy Boundary

Across problem families, follow this boundary:

- env:
  - owns dynamics
  - owns the canonical problem state/observation
  - returns raw state in a fixed format for that problem
- policy:
  - owns input normalization
  - owns the approximator
  - owns action decoding

Do not hide policy-side normalization inside the env. If a learned policy needs state scaling,
carry that explicitly as policy configuration and save it with the model and result metadata.

## Rust-First Folder Contract

For Rust-first additions, keep the executable module under `rust/src/problems/<problem>/` and put
the canonical artifact home under `rust/problems/<problem>/`.

Use this layout by default:

- `rust/problems/<problem>/`
  - `README.md`
  - `literature/`
  - `practical/`
    - `datasets/`
    - `reports/`
  - `experiments/`
  - `verification/`
- `rust/src/problems/<problem>/`
  - `env.rs`
  - `heuristics/`
    - `mod.rs`
    - one file per benchmark heuristic family when the module is more than trivial
  - `rollout.rs`
  - `references.rs`
  - `bindings.rs`
  - `mod.rs`
  - `tests/mod.rs`
  - `tests/verification.rs`

If literature-backed verification needs an exact finite-state solver or analytical evaluator, put it
in a role-specific module such as `finite_horizon_dp.rs`, `value_iteration_mdp.rs`, or
`rolling_scarf_dp.rs` and keep it separate from `heuristics/`.

Artifact responsibilities:

- `rust/problems/<problem>/README.md`
  - human-readable home for the family
  - points to code, literature notes, practical benchmarks, and verification targets
- `rust/problems/<problem>/literature/`
  - benchmark interpretation, source scope, and paper notes
- `rust/problems/<problem>/practical/datasets/`
  - checked-in practical dataset descriptors or trace files
- `rust/problems/<problem>/practical/reports/`
  - checked-in canonical report snapshots
- `rust/problems/<problem>/experiments/`
  - paper-facing experiment definitions and reported-instance selections
- `rust/problems/<problem>/verification/`
  - human-readable verification targets and semantics
- `scripts/<problem>/run_practical_benchmark.py`
  - executable entrypoint that refreshes the practical report

Common optional helper modules:

- `demand.rs`
- `allocation.rs`
- `supply.rs`
- `policies.rs`
- `finite_horizon_dp.rs`
- `value_iteration_mdp.rs`
- `rolling_scarf_dp.rs`

## Recommended First Tests

The first verification test for a new family should usually include:

- one reference-shape test
  - references exist and the canonical instance is wired correctly
- one state-layout test
  - policy features or state formatting are stable
- one worked-transition test
  - one-step inventory and cost accounting is correct
- one heuristic freeze test
  - benchmark heuristic action or parameter output is stable
- one exact-verifier test
  - exact policy or benchmark heuristic cost matches the frozen reference values

That is the minimum bar before we move to the next problem family.
