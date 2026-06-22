# Verification Target - nonstationary_lot_sizing

## Primary Target

| Field | Value |
| --- | --- |
| Status | `reference_companion_code_number` |
| Instance | `constant_10` author testbed row |
| Metric | total cost and shortage rate for simple `(s,S)` and rolling-DP `(s,S)` policies |
| Companion value | simple cost `1832.9142436489014`, simple shortage `0.0029443487165113735`; rolling-DP cost `1711.741`, rolling-DP shortage `0.04793465748308879` |
| Current repo value | Monte Carlo reproduction via `run_literature_benchmark.py` |
| Tolerance | use script tolerances; stochastic check should be run with `25000` replications for publication-grade validation |
| Last validated | `2026-06-22` |

## Source

Dehaybe, Catanzaro, and Chevalier (2024), "Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand", EJOR 314(2):433-445, DOI `10.1016/j.ejor.2023.10.007`, plus the author's public `HenriDeh/DRL_MMULS` `single-item` branch testbed CSVs.

The carried numeric rows are from the public companion-code CSVs, not peer-reviewed article table cells. The repo correctly keeps `literature_verified=false` on these instances under the strict rule.

## Validation Command

```bash
python scripts/nonstationary_lot_sizing/run_literature_benchmark.py \
  --instances constant_10 \
  --replications 25000
```

Quick smoke version:

```bash
python scripts/nonstationary_lot_sizing/run_literature_benchmark.py \
  --instances constant_10 \
  --replications 1000
```

## Notes

This is a useful reference-implementation benchmark, but not a strict literature table reproduction. Future upgrade path: obtain an article-printed per-instance value and add a deterministic or high-replication assertion against that value.
