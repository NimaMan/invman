# Verification Target - dual_sourcing

## Primary Target

| Field | Value |
| --- | --- |
| Status | `published_gap_reproduction` |
| Instance | `dual_l2_ce105` |
| Metric | Figure 9 optimality gaps above bounded-DP optimum |
| Literature value | capped dual index `0.00%`, tailored base-surge `0.06%`, dual index `0.11%`, single index `0.56%` |
| Current repo value | capped dual index `0.005815%`, tailored base-surge `0.061463%`, dual index `0.116371%`, single index `0.567514%` |
| Tolerance | `0.01` percentage points |
| Last validated | `2026-06-22` |

## Source

Gijsbrechts, Boute, Van Mieghem, and Zhang (2022), "Can Deep Reinforcement Learning Improve Inventory Management? Performance on Dual Sourcing, Lost Sales and Multi-Echelon Problems", Manufacturing & Service Operations Management, DOI `10.1287/msom.2021.1064`, Section 6.2 / Figure 9.

The paper reports gap percentages, not absolute costs. This file therefore verifies a published gap label against the repo's bounded-DP denominator.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
r = ir.dual_sourcing_reference_benchmark_summary(
    "dual_l2_ce105",
    inventory_lower=-12,
    inventory_upper=24,
    tolerance=1e-8,
    max_iterations=250,
    search_seed=123,
    search_horizon=6000,
    warm_up_periods_ratio=0.2,
)
expected = {
    "capped_dual_index": 0.00,
    "tailored_base_surge": 0.06,
    "dual_index": 0.11,
    "single_index": 0.56,
}
for h in r["heuristics"]:
    name = h["policy_name"]
    got = h["optimality_gap_pct"]
    print(name, got, "published", expected[name])
    assert abs(got - expected[name]) <= 0.01
PY
```

## Notes

Longer lead-time rows are slower because the bounded-DP state grows quickly. Use this `l_r=2` row as the fast canonical future-agent check.
