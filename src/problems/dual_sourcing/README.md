# Dual Sourcing

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "dual_sourcing",
  "status": "published_gap_reproduction",
  "instance": {
    "id": "dual_l2_ce105",
    "parameters": {
      "regular_lead_time": 2,
      "expedite_cost": 105
    }
  },
  "comparator": {
    "policy": "single_index",
    "metric": "optimality_gap_percentage_points"
  },
  "literature": {
    "value": 0.56,
    "units": "percentage points",
    "source": "Gijsbrechts et al. (2022), Can Deep Reinforcement Learning Improve Inventory Management?",
    "locator": "Section 6.2 / Figure 9, single-index row for dual_l2_ce105",
    "url_or_doi": "https://doi.org/10.1287/msom.2021.1064"
  },
  "reproduction": {
    "current_value": 0.567514,
    "tolerance": {
      "absolute_percentage_points": 0.01
    },
    "last_validated": "2026-06-22",
    "command": "python - <<'PY'\nimport invman_rust as ir\nr = ir.dual_sourcing_reference_benchmark_summary(\n    \"dual_l2_ce105\",\n    inventory_lower=-12,\n    inventory_upper=24,\n    tolerance=1e-8,\n    max_iterations=250,\n    search_seed=123,\n    search_horizon=6000,\n    warm_up_periods_ratio=0.2,\n)\nexpected = {\n    \"capped_dual_index\": 0.00,\n    \"tailored_base_surge\": 0.06,\n    \"dual_index\": 0.11,\n    \"single_index\": 0.56,\n}\nfor h in r[\"heuristics\"]:\n    name = h[\"policy_name\"]\n    got = h[\"optimality_gap_pct\"]\n    print(name, got, \"published\", expected[name])\n    assert abs(got - expected[name]) <= 0.01\nPY"
  }
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `published_gap_reproduction` |
| Instance | `dual_l2_ce105` |
| Metric | Figure 9 optimality gaps above bounded-DP optimum |
| Literature value | capped dual index `0.00%`, tailored base-surge `0.06%`, dual index `0.11%`, single index `0.56%` |
| Current repo value | capped dual index `0.005815%`, tailored base-surge `0.061463%`, dual index `0.116371%`, single index `0.567514%` |
| Tolerance | `0.01` percentage points |
| Last validated | `2026-06-22` |

### Source

Gijsbrechts, Boute, Van Mieghem, and Zhang (2022), "Can Deep Reinforcement Learning Improve Inventory Management? Performance on Dual Sourcing, Lost Sales and Multi-Echelon Problems", Manufacturing & Service Operations Management, DOI `10.1287/msom.2021.1064`, Section 6.2 / Figure 9.

The paper reports gap percentages, not absolute costs. This file therefore verifies a published gap label against the repo's bounded-DP denominator.

### Validation command

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

### Notes

Longer lead-time rows are slower because the bounded-DP state grows quickly. Use this `l_r=2` row as the fast canonical future-agent check.

This package implements the small-scale dual-sourcing benchmark family used by Gijsbrechts et al.
(2022).

## Formulation

- one regular supplier with lead time `l_r`
- one expedited supplier with lead time `l_e = 0`
- linear sourcing costs `c_r < c_e`
- end-of-period holding cost `h`
- end-of-period backlog cost `b`
- demand uniform on `{0, 1, 2, 3, 4}`

The package uses the six small benchmark settings carried by Gijs:

- `l_r in {2, 3, 4}`
- `c_e in {105, 110}`
- shared parameters `c_r = 100`, `h = 5`, `b = 495`

## Literature

Primary references:

- Gijsbrechts et al. (2022), M&SOM 24(3):1349-1368
- Veeraraghavan and Scheller-Wolf (2008), Operations Research 56(4):850-864
- Sheopuri et al. (2010), Operations Research 58(3):734-745

Executable literature references carried here:

- `dual_l2_ce105`
- `dual_l2_ce110`
- `dual_l3_ce105`
- `dual_l3_ce110`
- `dual_l4_ce105`
- `dual_l4_ce110`

Published benchmark comparators for this family are:

- `optimal_dp`
- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`
- `lp_adp`
- `a3c`

The public Gijs row-level numbers we carry for these settings are the Figure 9 optimality-gap
labels, not a full absolute-cost table.

## Heuristics

Implemented heuristic families:

- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`

Published benchmark takeaways:

- Gijs report that A3C is within `2%` of optimal on all six settings.
- Gijs show `capped_dual_index` as the strongest heuristic benchmark on this test family.
- Veeraraghavan and Scheller-Wolf are the source of the small-scale benchmark family.
- Sheopuri et al. extend the classical dual-sourcing policy family beyond the original dual-index
  rule.

## Verification

The package also carries a bounded dynamic-programming verifier for the six small settings.

This verifier is not a proof-level exact solver outside the chosen truncation box, so benchmark
claims should treat it as a bounded finite-state reference, not as a universal exact-optimality
claim for the unbounded model.

The repo already carries the six published Figure 9 optimality-gap labels in
`literature/references.rs`. Tests in `verification/tests.rs` freeze the full six-row table and
execute the canonical `dual_l2_ce105` benchmark instance against those labels.

## Policy Interface

The environment interface remains raw-state first:

- env/state builders expose raw reduced-state values
- normalization and feature construction happen in rollout or policy code, not inside the
  environment

## Benchmark Protocol

For this family, the literature-aligned comparison is:

- report repo heuristic or learned-policy costs against the bounded DP reference
- compare reproduced optimality gaps to the published Figure 9 gap labels

## Package Layout

- `experiments/`: paper-facing benchmark notes and future checked-in report snapshots
- `literature/`: literature anchors and the carried Figure 9 gap table
- `practical/`: practical benchmark placeholders when a canonical practical slice exists
- `verification/`: verification scope and the executable check strategy
