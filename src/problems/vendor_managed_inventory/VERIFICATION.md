# Verification Target - vendor_managed_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `handout_reference_not_peer_reviewed_repo_anchor` |
| Instance | Gosavi VMI worked newsvendor case |
| Metric | cycle demand order-up-to levels |
| Open reference value | mean-demand heuristic `15.0`, six-sigma `31.53`, displayed newsvendor `26.96` |
| Current repo value | mean-demand heuristic `15.0`, six-sigma `31.53122046311161`, newsvendor `26.9905428333404` |
| Tolerance | display rounding for `31.53` and `26.96`; exact for mean-demand `15.0` |
| Last validated | `2026-06-22` |

## Source

Gosavi teaching handout, "Case Study for Vendor-Managed Inventory (Based on Sui, Gosavi, & Lin, 2010)". This is an open instructional handout, not a peer-reviewed numeric benchmark. The peer-reviewed VMI paper's usable numeric table is not currently carried.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.vendor_managed_inventory_newsvendor_worked_case_summary()
print(s["mean_demand_heuristic_order_up_to"])
print(s["six_sigma_order_up_to"])
print(s["newsvendor_order_up_to"])
assert s["mean_demand_heuristic_order_up_to"] == 15.0
assert abs(s["six_sigma_order_up_to"] - 31.53) <= 0.01
assert abs(s["displayed_newsvendor_order_up_to"] - 26.96) <= 0.01
PY
```

## Notes

This file is intentionally conservative: the repo has a useful open worked-case anchor, but not a peer-reviewed literature number for the trainable VMI env. Upgrade only after locating and reproducing a citeable public row.
