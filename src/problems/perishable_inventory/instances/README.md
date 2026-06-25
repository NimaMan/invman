# Perishable Inventory Instances

This directory externalizes perishable Scenario A instances. It is the instance catalog for training and verification targets, not a benchmark-result report.

Published values are negative discounted returns, so higher is better. Farrington value-iteration values use the analytic midpoint-binned Gamma convention; rollout policies use sampled rounded demand, so do not mix estimators when making policy-improvement claims.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `de_moor2022_m2_exp3_l1_cp10_lifo.json` | `table_only` | Re-derivable small row, assertion not yet added |
| `de_moor2022_m3_exp5_l2_cp7_lifo.json` | `table_only` | Stored published value only |
| `de_moor2022_m4_exp5_l2_cp7_lifo.json` | `table_only` | Stored published value only |
| `de_moor2022_m5_exp8_l2_cp10_fifo.json` | `table_only` | Stored published value only |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test perishable_inventory::tests::verification -- --nocapture
```
