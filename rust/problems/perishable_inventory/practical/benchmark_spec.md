# Practical Benchmark Spec

Current dataset:

- `datasets/grocery_like_daily_trace.json`
- source kind: `repo_curated_semi_real`

Intent:

- expose a waste-service-cost tradeoff on a grocery-like daily demand block
- calibrate simple heuristics on a recent history block
- evaluate on a held-out block

Current protocol:

1. use the train block to estimate mean demand
2. search `base_stock`
3. search `bsp_low_ew`
4. evaluate both policies on:
   - the train block
   - the held-out test block

Reported metrics:

- mean period cost
- fill rate
- cycle service level
- waste rate relative to demand
- mean holding inventory
- mean order quantity
- positive-order frequency

Canonical runner:

- `scripts/perishable_inventory/run_practical_benchmark.py`

Latest checked-in report:

- `reports/latest_report.md`
- `reports/latest_report.json`
