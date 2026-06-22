# Verification Target - procurement_removal_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `no_public_literature_number_repo_exact_anchor` |
| Instance | reduced exact verification instance |
| Metric | finite-horizon discounted optimal cost |
| Literature value | none currently available |
| Current repo value | `31.78026111369698` |
| Tolerance | `1e-9` against the repo exact DP anchor |
| Last validated | `2026-06-22` |

## Source

Maggiar and Sadighian (2017), "Joint Inventory and Revenue Management with Removal Decisions", SSRN/Amazon Science working paper, is a structural source for the problem class. The repo does not currently carry a public control-only per-instance cost row from that paper.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.procurement_removal_inventory_exact_dp_summary()
print(s["optimal_discounted_cost"])
print(s["optimal_first_action"])
assert abs(s["optimal_discounted_cost"] - 31.78026111369698) <= 1e-9
assert list(s["optimal_first_action"]) == [0, 0]
PY
```

## Notes

This is a strong repo-native regression target, not a literature verification. Future upgrade path: identify a public procurement/removal instance with a printed or companion-code optimal value and add it here.
