# Practical Benchmark Spec

Current dataset:

- `datasets/retail_like_weekly_trace.json`
- source kind: `repo_curated_semi_real`

Intent:

- evaluate forecast-driven replenishment policies on one rolling forecast path
- measure operational behavior beyond one benchmark mean-cost row

Current protocol:

1. keep one fixed forecast-plus-realized-demand path
2. evaluate:
   - `lead_time_base_stock`
   - `simple_s_s`
   - `rolling_dp_s_s`
3. report the first-period levels for the `(s,S)` heuristics and operational metrics over the full
   path

Reported metrics:

- mean period cost
- shortage rate
- cycle service level
- mean holding inventory
- mean order quantity
- positive-order frequency

Canonical runner:

- `scripts/nonstationary_lot_sizing/run_practical_benchmark.py`

Latest checked-in report:

- `reports/README.md`
- `reports/latest_report.json`
