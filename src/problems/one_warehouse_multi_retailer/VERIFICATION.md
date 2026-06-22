# Verification Target - one_warehouse_multi_retailer

## Primary Target

| Field | Value |
| --- | --- |
| Status | `published_heuristic_simulation_match` |
| Instance | `kaynov2024_instance_7` |
| Metric | lost-sales echelon-base-stock min-shortage benchmark cost |
| Literature value | `1408.08` cost, standard error `0.95` |
| Current repo value | `1394.8165` cost with `1000` replications and seed `2222` |
| Tolerance | `1.2%` relative gap for this stochastic reproduction |
| Last validated | `2026-06-22` |

## Source

Kaynov et al. (2024), International Journal of Production Economics 267, 109088, DOI `10.1016/j.ijpe.2023.109088`, Table 1 / Table A.3 as carried in `references.rs`.

## Validation Command

```bash
python scripts/one_warehouse_multi_retailer/validate_reference_instance.py \
  --reference_name kaynov2024_instance_7 \
  --benchmark_replications 1000 \
  --seed 2222
```

Expected output includes:

```text
echelon_base_stock_min_shortage published 1408.080 repo 1394.816 relative gap -0.942%
```

## Notes

The sign convention differs between the paper-carried reward/profit values and this validation script's positive cost display. This file uses positive cost magnitudes. The full problem remains hard: only a subset of Kaynov rows reproduce tightly, and the exact DP anchor is repo-native rather than a full 100-period literature optimum.
