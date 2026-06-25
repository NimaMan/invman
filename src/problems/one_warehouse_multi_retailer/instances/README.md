# One-Warehouse Multi-Retailer Instances

This directory is the machine-readable instance catalog for the one-warehouse multi-retailer problem. Lost-sales rows are intentionally excluded from this expansion pass because the lost-sales family already has enough coverage.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `gijsbrechts2022_table3_setting2_mu0_k10_partial_backorder.json` | `faithful_unverified` | Literature-shaped partial-backorder row, no absolute OWMR value |
| `van_roy1997_case2_calibrated_mu1_k10_partial_backorder.json` | `faithful_unverified` | Calibrated reproduction target, not a strict OWMR table row |
| `generated_backorder_k5_heterogeneous_demand_costs.json` | `generated` | Synthetic stress case |
| `generated_partial_backorder_k10_leadtime_gradient.json` | `generated` | Synthetic stress case |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test -q one_warehouse_multi_retailer::tests::verification -- --nocapture
```
