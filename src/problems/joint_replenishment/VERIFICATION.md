# Verification Target - joint_replenishment

## Primary Target

| Field | Value |
| --- | --- |
| Status | `published_action_reproduction` |
| Instance | Vanvuchelen et al. setting 5 |
| Metric | optimal action at state `(I1, I2) = (5, 0)` |
| Literature value | `q = (0, 6)` |
| Current repo value | `q = (0, 6)` via independent infinite-horizon VI script |
| Tolerance | exact action match |
| Last validated | `2026-06-22` |

## Source

Vanvuchelen, Gijsbrechts, and Boute (2020), "Use of Proximal Policy Optimization for the Joint Replenishment Problem", Computers in Industry 119, 103239, DOI `10.1016/j.compind.2020.103239`, Figure 3 / setting 5.

The published anchor is an action, not an absolute cost. The paper does not provide a public absolute optimal-cost table for this setting.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
a = ir.joint_replenishment_published_action_anchor()
print(a)
assert list(a["state_inventory_levels"]) == [5, 0]
assert list(a["optimal_action"]) == [0, 6]
PY
```

For a full re-derivation rather than inspecting the carried anchor:

```bash
python scripts/joint_replenishment/benchmark_vanvuchelen_settings.py --periods 1 --replications 1
```

## Notes

The reduced finite-horizon DP summary in `joint_replenishment_exact_dp_summary()` is a repo self-consistency comparator; it should not be confused with the Figure 3 literature action.
