# perishable_inventory

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "perishable_inventory",
  "instance_id": "de_moor2022_m2_exp2_l1_cp7_fifo",
  "instance_parameters": {
    "lifetime": 2,
    "lead_time": 1,
    "policy": "FIFO"
  },
  "policy": "value_iteration",
  "metric": "mean_return_rounded",
  "expected_value": -1457,
  "reference": {
    "citation": "Farrington, Wong, Li, and Utley (2025), Going faster to see further: graphics processing unit-accelerated value iteration and simulation for perishable inventory control using JAX",
    "locator": "Table 3, m=2, experiment 2, L=1, c_p=7, FIFO row",
    "doi_or_url": "https://doi.org/10.1007/s10479-025-06551-6",
    "literature_verified": true,
    "notes": "Rounded mean-return table convention; De Moor et al. (2022) is the secondary structural source."
  },
  "code_value": -1457.281304782201,
  "tolerance": {
    "rounded_exact": true
  },
  "command": "python - <<'PY'\nimport invman_rust as ir\ns = ir.perishable_inventory_exact_mdp_summary(\"de_moor2022_m2_exp2_l1_cp7_fifo\")\nprint(s[\"value_iteration_mean_return\"])\nprint(s[\"value_iteration_mean_return_rounded\"])\nprint(s[\"matches_published_value_iteration_mean_return\"])\nprint(s[\"matches_published_policy_table\"])\nprint(s[\"matches_published_base_stock_level\"])\nassert s[\"value_iteration_mean_return_rounded\"] == -1457\nassert s[\"matches_published_value_iteration_mean_return\"]\nassert s[\"matches_published_policy_table\"]\nassert s[\"matches_published_base_stock_level\"]\nPY"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | `de_moor2022_m2_exp2_l1_cp7_fifo` |
| Metric | value-iteration mean return, rounded to article table convention |
| Literature value | `-1457` |
| Current repo value | `-1457.281304782201`, rounded `-1457` |
| Tolerance | exact rounded match |
| Last validated | `2026-06-22` |

### Source

Farrington, Wong, Li, and Utley (2025), "Going faster to see further: graphics processing unit-accelerated value iteration and simulation for perishable inventory control using JAX", Annals of Operations Research 349(3):1609-1638, Table 3, DOI `10.1007/s10479-025-06551-6`.

Secondary structural source: De Moor, Gijsbrechts, and Boute (2022), "Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management", European Journal of Operational Research 301(2):535-545, Figure 3 policy tables and base-stock levels, DOI `10.1016/j.ejor.2021.10.045`.

### Validation command

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

### Notes

This is a true exact-MDP reproduction on a tractable `m=2`, `L=1` slice. Larger perishable instances may be table-only or practical-benchmark targets.

Canonical Rust-first home for the perishable-inventory family.

Code:

- implementation: `src/problems/perishable_inventory/`
- tests: `src/problems/perishable_inventory/tests/verification.rs`

Artifact folders:

- `literature/`
  - paper scope and benchmark interpretation
- `practical/`
  - checked-in practical trace, benchmark spec, and latest report snapshot
- `experiments/`
  - paper-facing benchmark definition
- `verification/`
  - human-readable statement of what the exact verifier asserts

Verification status:

- LITERATURE-VERIFIED on the `m = 2`, lead-time-1 slice ONLY (four 121-state
  instances). The exact value-iteration MDP re-derives, in-repo at test time, the
  De Moor et al. (2022, EJOR 301(2):535-545) optimal-policy tables and best
  base-stock levels (5 LIFO, 7 FIFO) and the Farrington, Wong, Li, Utley (2025,
  Ann. Oper. Res. 349(3):1609-1638) Table 3 value-iteration returns (-1553 LIFO,
  -1457 FIFO) exactly. The repo labels the De Moor policy tables "Figure 3"; the
  exact published figure number was not independently confirmed (paywalled EJOR
  full text). The other 28 Scenario A rows are TABLE-ONLY anchors (stored, not
  re-derived). Details, citation-correctness notes, and the estimator caveat are
  in `literature/README.md`.

Current anchors:

- primary literature instance: `de_moor2022_m2_exp2_l1_cp7_fifo`
- exact verification instances:
  - `de_moor2022_m2_exp1_l1_cp7_lifo`
  - `de_moor2022_m2_exp2_l1_cp7_fifo`
- practical benchmark instance: `de_moor2022_m4_exp6_l2_cp7_fifo`

Benchmark:

- working runner: `scripts/perishable_inventory/run_exact_slice_benchmark.py`
  (exact optimum vs tuned `base_stock` / `bsp_low_ew` vs CMA-ES soft tree)
- latest report: `experiments/reports/exact_slice_report/README.md`
- NOTE: the older `scripts/perishable_inventory/run_paper_benchmark.py` is dead
  (imports the removed `invman.policies.soft_tree`); use the runner above.

State interface:

- `env.rs` exposes raw inventory and pipeline quantities in observation order
- any scaling used by a learned policy belongs in `rollout.rs` or the policy itself
- environment code must not silently normalize policy inputs
