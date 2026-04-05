# Problems

Each inventory-control problem lives in its own subpackage under `invman/problems` or, for newer
families, under `rust/src/problems/` first.

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

The first four have Python problem packages today. The later six are currently Rust-first.

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

Learned policy classes stay separate under `invman/policies/`. The problem packages own the
simulation, baseline heuristics, and reference benchmarks.

## Current Direction

New problem families are Rust-first.

That means:

- the canonical first implementation lives under `rust/src/problems/<problem>/`
- Python can mirror the structure later if we need a higher-level package wrapper
- the Rust module must already contain the environment, heuristic baselines, rollout path, and
  verification anchors before the family counts as implemented

The newer Rust-first families currently include the six families listed above.

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

For Rust-first additions, the same concepts should live under `rust/src/problems/<problem>/`
first, and the Python package can mirror that structure later.

Use this layout by default:

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
