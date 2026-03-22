# Problems

Each inventory-control problem lives in its own subpackage under `invman/problems`.

- `lost_sales/`
  - `env.py`: simulator and rollout helpers
  - `heuristics.py`: classic baseline policies
  - `reference_instances.py`: trusted benchmark instances
  - `problem_info.py`: literature reference tables
  - `benchmark.py`: default heuristic evaluation helpers
- `lost_sales_fixed_order_cost/`
  - `env.py`: fixed-cost problem entrypoint built on the lost-sales simulator
  - `heuristics.py`: `(s,S)`, `(s,nQ)`, modified `(s,S,q)` search and evaluation
  - `reference_instances.py`: literature-derived benchmark grid
  - `benchmark.py`: benchmark runners for the fixed-cost grid

Learned policy classes stay separate under `invman/policies/`. The problem packages own the
simulation, baseline heuristics, and reference benchmarks.
