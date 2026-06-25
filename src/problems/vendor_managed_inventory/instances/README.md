# Vendor Managed Inventory Instances

This directory contains machine-readable instances for `src/problems/vendor_managed_inventory`.

Current caveat: no VMI instance is `strict_literature`. The open executable anchor is Gosavi's worked newsvendor handout based on Sui, Gosavi and Lin (2010). The peer-reviewed 8-case profit rows are carried only as `table_only` because the public implementation does not reproduce them. Reduced single-retailer instances are repo-native comparison and stress instances.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `gosavi2010_worked_newsvendor_case.json` | `companion_code` | Handout worked example reproduced |
| `sui_gosavi_lin_2010_reduced_single_retailer_primary.json` | `faithful_unverified` | Repo-native reduced benchmark |
| `vmi_reduced_high_penalty.json` | `generated` | Generated perturbation |
| `sui_gosavi_lin_2010_truck_dispatch_case01.json` | `table_only` | Paper table row, not reproduced |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test -q vendor_managed_inventory::verification --lib
```
