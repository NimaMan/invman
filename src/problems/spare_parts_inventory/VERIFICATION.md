# Verification Target - spare_parts_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `reference_adjacent_strict_number` |
| Instance | `kranenburg2006_table5_2_base_case` |
| Metric | Kranenburg lateral-transshipment analytical costs |
| Literature value | situation 1: `R*=9.09`, `C=91.90`; situation 3: `R*=6.10`, `C=63.00`; ratio `1.46` |
| Current repo value | situation 1: `R=9.09`, `C=91.90`; situation 3: `R=6.100000033871978`, `C=63.00000032739136`; ratio `1.458730151149593` |
| Tolerance | `0.02` absolute table-rounding tolerance |
| Last validated | `2026-06-22` |

## Source

Kranenburg (2006), "Spare parts inventory control under system availability constraints", PhD thesis, Technische Universiteit Eindhoven, Chapter 5, Tables 5.1-5.3, DOI `10.6100/IR616052`.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.spare_parts_inventory_kranenburg_exact_summary("kranenburg2006_table5_2_base_case")
print(s["evaluation"])
print(s["published_table_comparison"])
assert s["published_table_comparison"]["all_within_tolerance"]
PY
```

## Notes

This strict number is for an adjacent analytical lateral-transshipment module, not the trainable periodic-review repairable-spares env. The trainable env still uses `spare_parts_inventory_exact_dp_summary()` as a repo-native exact anchor.
