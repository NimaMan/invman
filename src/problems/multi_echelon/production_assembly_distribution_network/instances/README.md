# Production Assembly Distribution Network Instances

This directory contains machine-readable PADN/Pirhooshyaran-Snyder network instances. These are not the Rosling `multi_echelon/assembly` instances; relation order is internal edges in listed order, then external supplier relations by source-node index.

Published table rows are `table_only` until a live PADN env/solver re-runs the reported cost under the stated protocol. Serial rows from the same paper are strict in `multi_echelon/serial`, but only table-only when represented through PADN.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `pirhooshyaran2021_assembly1_case2.json` | `table_only` | Published PADN table row, not rerun |
| `pirhooshyaran2021_assembly2_case5.json` | `table_only` | Published PADN table row, not rerun |
| `pirhooshyaran2021_mixed_fig1_table7.json` | `table_only` | Published mixed network row, not rerun |
| `pirhooshyaran2021_serial_case1.json` | `table_only` | Strict home is `multi_echelon/serial` |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python -c "import invman_rust as ir,json; print(json.dumps(ir.production_assembly_distribution_network_literature_benchmark_summary(serial_replications=10000,seed=1234),default=str))"
```
