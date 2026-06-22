# Verification Target - ameliorating_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `reference_companion_bound` |
| Instance | `pahr_grunow2025_spirits_0001` |
| Metric | perfect-information LP upper bound on average profit, `max_reward` |
| Literature / companion value | `1991.9344293376805` |
| Current repo value | `1991.9344293930808` |
| Tolerance | `1e-6` absolute |
| Last validated | `2026-06-22` |

## Source

Pahr and Grunow (2025), "The Value of Blending - Managing Ameliorating Inventory Using Deep Reinforcement Learning", Production and Operations Management, DOI `10.1177/10591478251387795`.

The numeric anchor is the public companion-code/data LP bound carried in `references.rs`, not an achieved trainable-env policy value. The trainable environment is still faithful/unverified against a published achieved-cost or achieved-profit number.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.ameliorating_inventory_perfect_info_lp_bound_summary("pahr_grunow2025_spirits_0001")
print(s["upper_bound_max_reward"])
print(s["published_max_reward"])
print(s["max_reward_gap_to_published"])
assert abs(s["upper_bound_max_reward"] - s["published_max_reward"]) <= 1e-6
PY
```

## Notes

Use this as an upper-bound reproduction check only. A learned policy can be compared to the bound as a gap-to-bound result, but matching or beating the bound is not the right expectation because it is perfect-information.
