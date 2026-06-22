# Verification Target - joint_pricing_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `no_public_literature_number_repo_exact_anchor` |
| Instance | reduced exact verification instance |
| Metric | finite-horizon discounted optimal cost |
| Literature value | none currently available |
| Current repo value | `-33.178121049724` |
| Tolerance | `1e-9` against the repo exact DP anchor |
| Last validated | `2026-06-22` |

## Source

The code cites formulation-class references such as Qin, Simchi-Levi, and Wang (2022), DOI `10.1287/mnsc.2021.4212`, but the repo does not currently carry a public per-instance optimal-profit or optimal-cost number from the literature for this exact reduced MDP.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.joint_pricing_inventory_exact_dp_summary()
print(s["optimal_discounted_cost"])
print(s["optimal_first_action"])
assert abs(s["optimal_discounted_cost"] - (-33.178121049724)) <= 1e-9
assert list(s["optimal_first_action"]) == [2, 1]
PY
```

## Notes

This file intentionally records the absence of a literature number. Future upgrade path: find a citeable joint-pricing-and-inventory worked instance with a public optimal value, add it to `literature/references.rs`, and replace this repo-native anchor with the published number.
