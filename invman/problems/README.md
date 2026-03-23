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
