# Verification Target - perishable_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | `de_moor2022_m2_exp2_l1_cp7_fifo` |
| Metric | value-iteration mean return, rounded to article table convention |
| Literature value | `-1457` |
| Current repo value | `-1457.281304782201`, rounded `-1457` |
| Tolerance | exact rounded match |
| Last validated | `2026-06-22` |

## Source

Farrington, Wong, Li, and Utley (2025), "Going faster to see further: graphics processing unit-accelerated value iteration and simulation for perishable inventory control using JAX", Annals of Operations Research 349(3):1609-1638, Table 3, DOI `10.1007/s10479-025-06551-6`.

Secondary structural source: De Moor, Gijsbrechts, and Boute (2022), "Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management", European Journal of Operational Research 301(2):535-545, Figure 3 policy tables and base-stock levels, DOI `10.1016/j.ejor.2021.10.045`.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.perishable_inventory_exact_mdp_summary("de_moor2022_m2_exp2_l1_cp7_fifo")
print(s["value_iteration_mean_return"])
print(s["value_iteration_mean_return_rounded"])
print(s["matches_published_value_iteration_mean_return"])
print(s["matches_published_policy_table"])
print(s["matches_published_base_stock_level"])
assert s["value_iteration_mean_return_rounded"] == -1457
assert s["matches_published_value_iteration_mean_return"]
assert s["matches_published_policy_table"]
assert s["matches_published_base_stock_level"]
PY
```

## Notes

This is a true exact-MDP reproduction on a tractable `m=2`, `L=1` slice. Larger perishable instances may be table-only or practical-benchmark targets.
