# Procurement Removal Inventory Instances

This directory is the instance catalog for `procurement_removal_inventory`. Current strict literature and companion-code count is zero: the rows here are faithful repo/native anchors, table-only paper context, or generated stress cases.

Two model variants are represented:

- `control_only_procurement_removal`: finite-horizon order/remove control without pricing.
- `faithful_pricing_backlog_gamma`: closer to the Maggiar-Sadighian pricing/backlogging model, but not a tight executable reproduction of a printed value.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `reduced_exact_verification_instance.json` | `faithful_unverified` | Exact DP self-consistency |
| `maggiar2017_style_fixed_returnability.json` | `faithful_unverified` | Repo-native MC benchmark |
| `maggiar_sadighian_2017_table1_full_model.json` | `table_only` | Approximate published-figure context |
| `generated_full_returnability_overstock.json` | `generated` | Synthetic stress case |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python -c "import invman_rust as r; print(r.procurement_removal_inventory_exact_dp_summary())"
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py
```
