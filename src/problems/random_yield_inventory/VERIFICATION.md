# Verification Target - random_yield_inventory

## Primary Target

| Field | Value |
| --- | --- |
| Status | `no_public_literature_number_repo_exact_anchor` |
| Instance | reduced exact verification instance |
| Metric | finite-horizon discounted optimal cost |
| Literature value | none currently available for this exact all-or-nothing yield MDP |
| Current repo value | `40.05989760985441` |
| Tolerance | `1e-9` against the repo exact DP anchor |
| Last validated | `2026-06-22` |

## Source

Yan, Chen, Fu, and Bi (2026), "Heuristics and deep reinforcement learning for the inventory problem with an all-or-nothing yield pattern and non-zero leadtimes", Computers & Operations Research 186, 107305, DOI `10.1016/j.cor.2025.107305`, is the current formulation-class source. The repo has not yet found a public per-instance numeric target from that paper or a matching earlier paper.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.random_yield_inventory_exact_dp_summary()
print(s["optimal_discounted_cost"])
print(s["optimal_first_action"])
assert abs(s["optimal_discounted_cost"] - 40.05989760985441) <= 1e-9
assert s["optimal_first_action"] == 4
PY
```

## Notes

Do not claim literature verification from this file. It exists so future agents can still check that the repo's random-yield mechanics have not drifted while searching for a public numeric literature anchor.
