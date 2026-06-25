# General Backorder Instances

This directory is the machine-readable instance catalog for `multi_echelon/general_backorder_fixed_cost`.

The family name is historical: the currently verified Geevers/Kunnumkal-Topaloglu rows are holding-plus-backorder benchmarks, not fixed-order-cost reproductions. `geevers2023_general_set2` and `geevers2023_general_set3` are deliberately stored as `table_only`; local simulation still misses the gated order-per-edge transition and must not report those rows as reproduced.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `geevers2023_general_set1.json` | `strict_literature` | Reproduced by benchmark script within tolerance |
| `kunnumkal_topaloglu_divergent.json` | `strict_literature` | Reproduced by benchmark script within tolerance |
| `geevers2023_general_set2.json` | `table_only` | Known unreproduced |
| `geevers2023_general_set3.json` | `table_only` | Known unreproduced |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test general_backorder_fixed_cost --lib
python scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py
```
