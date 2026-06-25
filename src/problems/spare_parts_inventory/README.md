# spare_parts_inventory

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "spare_parts_inventory",
  "instance_id": "kranenburg2006_table5_2_base_case",
  "instance_parameters": {
    "situation": 1
  },
  "policy": "lateral_transshipment_analytical",
  "metric": "cost",
  "expected_value": 91.9,
  "reference": {
    "citation": "Kranenburg (2006), Spare parts inventory control under system availability constraints",
    "locator": "Chapter 5, Table 5.2 base case, situation 1 cost C",
    "doi_or_url": "https://doi.org/10.6100/IR616052",
    "literature_verified": true,
    "notes": "Adjacent analytical lateral-transshipment module; this does not verify the trainable periodic-review env.rs model."
  },
  "code_value": 91.9,
  "tolerance": {
    "absolute": 0.02
  },
  "command": "python - <<'PY'\nimport invman_rust as ir\ns = ir.spare_parts_inventory_kranenburg_exact_summary(\"kranenburg2006_table5_2_base_case\")\nprint(s[\"evaluation\"])\nprint(s[\"published_table_comparison\"])\nassert s[\"published_table_comparison\"][\"all_within_tolerance\"]\nPY"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `reference_adjacent_strict_number` |
| Instance | `kranenburg2006_table5_2_base_case` |
| Metric | Kranenburg lateral-transshipment analytical costs |
| Literature value | situation 1: `R*=9.09`, `C=91.90`; situation 3: `R*=6.10`, `C=63.00`; ratio `1.46` |
| Current repo value | situation 1: `R=9.09`, `C=91.90`; situation 3: `R=6.100000033871978`, `C=63.00000032739136`; ratio `1.458730151149593` |
| Tolerance | `0.02` absolute table-rounding tolerance |
| Last validated | `2026-06-22` |

### Source

Kranenburg (2006), "Spare parts inventory control under system availability constraints", PhD thesis, Technische Universiteit Eindhoven, Chapter 5, Tables 5.1-5.3, DOI `10.6100/IR616052`.

### Validation command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.spare_parts_inventory_kranenburg_exact_summary("kranenburg2006_table5_2_base_case")
print(s["evaluation"])
print(s["published_table_comparison"])
assert s["published_table_comparison"]["all_within_tolerance"]
PY
```

### Notes

This strict number is for an adjacent analytical lateral-transshipment module, not the trainable periodic-review repairable-spares env. The trainable env still uses `spare_parts_inventory_exact_dp_summary()` as a repo-native exact anchor.

Rust-first problem home for `spare_parts_inventory`.

Repo interpretation:

- repairable spare-parts control
- installed-base failures create demand
- procurement and repair pipelines jointly determine service and downtime
- this folder may also catalog adjacent spare-parts literature benchmarks when a paper publishes
  reusable numeric benchmark rows, even if the repo-native executable primary instance is a
  different spare-parts subfamily

## Verification status (honest, per docs/rust/README.md "What counts as literature-verified")

A benchmark is literature-verified ONLY when an in-crate test RE-RUNS the env/solver
and asserts the freshly computed metric reproduces a number PRINTED IN A PAPER within
a stated tolerance. By that rule:

- LITERATURE-VERIFIED (and the ONLY one in this family): Kranenburg (2006) Chapter 5
  Table 5.2 lateral-transshipment comparison. The ANALYTICAL module
  `literature/kranenburg_lateral_transshipment.rs` re-derives R* and total cost for
  Situation 1 (separate stock points) and Situation 3 (lateral transshipment) and the
  test `kranenburg_table_5_2_rows_are_reproduced_within_table_rounding` reproduces every
  printed Table 5.2 row (Kranenburg 2006 PhD thesis, TU/e, Chapter 5, p.107) within
  tolerance 0.02. This is a CONTINUOUS-REVIEW, METRIC-style multi-location model and is
  STRUCTURALLY A DIFFERENT MODEL from the trainable `env.rs`. Its verification covers the
  analytical module only.

- NOT literature-verified: the trainable environment `env.rs` (the repo-native
  single-echelon PERIODIC-REVIEW repairable MDP: binomial failures, deterministic repair
  return after `repair_lead_time`, backorders, order-after-demand). No paper publishes a
  numeric cost for this exact construction. Its tests are a characterization / drift guard
  (`env_periodic_review_trajectory_is_pinned_characterization_not_literature`,
  `worked_transition_matches_expected_accounting`) and a self-consistency DP comparison
  (`exact_dp_dominates_repo_heuristics`) -- none reproduces a paper number.
  `references.rs` flags both `PRIMARY_REFERENCE_INSTANCE` and
  `VERIFICATION_PROBLEM_INSTANCE` with `literature_verified = false`.

- NOT literature-verified: van Oers et al. (2024) Table 1 two-echelon serial benchmark.
  The table values are RECORDED constants only; no env/solver here re-runs them, so the
  test `van_oers_2024_table_is_recorded_but_not_literature_verified` is a frozen snapshot,
  which the repo rule excludes from "verified". Flagged `literature_verified = false`.
  Kept as a catalog target for a future executable two-echelon serial env.

DO NOT let the analytical Kranenburg numbers imply env.rs is verified -- they describe a
different model.

Code lives under `src/problems/spare_parts_inventory/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface rule:

- `env.rs` exposes raw state quantities only
- any normalization, scaling, or derived inventory-position features for learned policies must live outside the environment layer
- `rollout.rs` is the right place to convert raw state into the feature vector expected by a specific policy family
