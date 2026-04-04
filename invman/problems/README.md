# Problems

Each inventory-control problem lives in its own subpackage under `invman/problems`.

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

Learned policy classes stay separate under `invman/policies/`. The problem packages own the
simulation, baseline heuristics, and reference benchmarks.

## Standard For New Problems

The current four packages were added incrementally and are not fully uniform. New problem families
should follow one standard from the start.

Required artifacts for every new problem family:

- `README.md`: short literature note, benchmark scope, and the canonical repo interpretation of the
  problem family
- `reference_instances.py` or the Rust-first equivalent source of truth:
  - all literature instances we want to carry forward from the papers
  - one `PRIMARY_REFERENCE_INSTANCE`
  - one `VERIFICATION_PROBLEM_INSTANCE`
  - published numbers when they exist
  - explicit notes when repo values are repo-native rather than verbatim literature values
- `heuristics.py` or Rust-first equivalent:
  - the classical benchmark policies for that family
  - search helpers if the heuristic requires parameter tuning
- `env.py` or Rust-first equivalent:
  - state transition logic and cost accounting
- `benchmark.py` when we have a nontrivial benchmark grid
- tests tied to the verification instance

Required semantics:

- the references file is the authoritative list of literature instances used in the repo
- the verification instance is the minimal correctness anchor for environment dynamics and
  heuristic behavior
- the primary reference instance is the canonical first benchmark for learned policies

For Rust-first additions, the same concepts should live under `rust/src/problems/<problem>/`
first, and the Python package can mirror that structure later.

For Rust-first problem families, use this layout by default:

- `env.rs`
- `heuristics/`
  - `mod.rs`
  - one file per benchmark heuristic family when the module is more than trivial
- `rollout.rs`
- `references.rs`
- `bindings.rs`
- `tests/verification.rs`

If literature-backed verification needs an exact finite-state solver or analytical evaluator, put it
in `exact.rs` and keep it separate from `heuristics/`.
