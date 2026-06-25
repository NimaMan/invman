# Spare Parts Inventory Instances

This directory externalizes reusable spare-parts problem instances. It is an instance catalog, not a benchmark-result report.

Kranenburg 2006 rows are strict literature for an adjacent continuous-review lateral-transshipment model. They do not verify the trainable periodic-review repairable `env.rs`. van Oers 2024 rows are table-only until a two-echelon serial-AM solver is implemented. The single-echelon repairable instances are repo-native and must remain `faithful_unverified`.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `kranenburg2006_table5_2_base_case.json` | `strict_literature` | Analytical Kranenburg verifier |
| `kranenburg2006_table5_2_m_0p01.json` | `strict_literature` | Analytical Kranenburg verifier |
| `van_oers2024_table1_no_am.json` | `table_only` | Frozen table snapshot only |
| `single_echelon_repairable_exact_dp_reduced.json` | `faithful_unverified` | Repo-native exact DP |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python scripts/spare_parts_inventory/validate_kranenburg_lateral_transshipment.py --all_rows
python scripts/spare_parts_inventory/validate_against_exact_dp.py
cargo test spare_parts_inventory::tests::verification -- --nocapture
```
