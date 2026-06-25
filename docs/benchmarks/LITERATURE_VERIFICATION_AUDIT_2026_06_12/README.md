# Literature-verification audit — 2026-06-12

Adversarial per-family audit (14 families × an audit agent + an independent
refute agent = 28 agents) of which `invman` problems are genuinely
**literature-verified** — i.e. the env reproduces a number that EXISTS in the
literature — versus repo-native `<author>_style` instances that an exact solver
happens to solve. Run id `wf_a99686b4-f99`. No verdict was demoted by the refuter.

**Definitions.** `strict` = re-runs a number **printed in a peer-reviewed**
paper/table within tolerance. `reference` = re-runs a **companion-code /
closed-form / reduced-module** number, or a **published action** (not a printed
cost). `faithful` = **no public anchor**; a repo-native / inspired-by instance
validated only against the repo's own DP — **NOT** literature-verified.

## Verdict: 9 of 14 literature-verified, 5 not

| Family | Tier | Anchor type | Literature-verified |
| --- | --- | --- | :---: |
| `lost_sales` | strict | peer_reviewed_printed_number | ✅ |
| `perishable_inventory` | strict | peer_reviewed_printed_number | ✅ |
| `spare_parts_inventory` | mixed | mixed | ✅ |
| `multi_echelon` | mixed | mixed | ✅ |
| `dual_sourcing` | reference | published_action | ✅ |
| `joint_replenishment` | reference | published_action | ✅ |
| `nonstationary_lot_sizing` | reference | companion_code_number | ✅ |
| `ameliorating_inventory` | mixed | companion_code_number | ✅ |
| `decentralized_inventory_control` | mixed | closed_form_number | ✅ |
| `one_warehouse_multi_retailer` | faithful | repo_native_style_no_anchor | ❌ (repo-native) |
| `joint_pricing_inventory` | faithful | closed_form_number | ❌ (repo-native) |
| `procurement_removal_inventory` | faithful | repo_native_style_no_anchor | ❌ (repo-native) |
| `random_yield_inventory` | faithful | repo_native_style_no_anchor | ❌ (repo-native) |
| `vendor_managed_inventory` | faithful | repo_native_style_no_anchor | ❌ (repo-native) |

`available_runners()` and `catalog.list_problems(literature_verified=True)`
default to the 9 verified families; the 5 faithful ones stay reachable via
`get_runner(...)` / `load_instance(...)` and `include_unverified=True`.

## Per-family evidence

### `lost_sales` — strict

- **Anchor:** peer_reviewed_printed_number; reproduces published: True.
- **Evidence:** Both audited subfamilies anchor to PEER-REVIEWED PRINTED table numbers and RE-RUN them within tolerance, so STRICT. VANILLA (Zipkin 2008, Operations Research 56(5):1256-1263, Table 3(a), Poisson mu=5, L=4, h=1, p=4): - src/problems/lost_sales/vanilla/literature/references.rs:70-116 carries the printed rows Optimal 4.73, Myopic 5.06, Myopic-2 4.82, SVBS 5.83, Better-VBS 4.80;
- **Caveat:** Per-subfamily both STRICT, but with scoped caveats on which rows are executable vs carried: VANILLA (strict): Only the canonical instance vanilla_l4_p4_poisson5 (alias lit_poisson_p4_l4) is strict-verified, and only its 3 HEURISTIC rows (myopic1 5.06, myopic2 4.82, svbs 5.83) are re-run by the live env. The published OPTIMAL 4.73 and Better-VBS/capped 4.80 rows are TABLE-ONLY c…
- **Adversarial refute:** verdict held; Verdict holds as STRICT.

### `perishable_inventory` — strict

- **Anchor:** peer_reviewed_printed_number; reproduces published: True.
- **Evidence:** VERDICT: STRICT. The env re-derives numbers PRINTED in two PEER-REVIEWED journal tables, within tolerance, with COMPUTED (not snapshot) values.
- **Caveat:** Single subfamily (de_moor2022_scenario_a), so tier is NOT mixed -- it is uniformly strict at the verified-instance level. Honest scope limits: 1.
- **Adversarial refute:** verdict held; The strict verdict survives adversarial scrutiny;

### `spare_parts_inventory` — mixed

- **Anchor:** mixed; reproduces published: True.
- **Evidence:** The family is HONESTLY SPLIT into three sub-models with different verdicts; the repo is scrupulous about this.
- **Caveat:** Per-subfamily tiers: 1) kranenburg_2006_lateral_transshipment_analytical = STRICT. Re-runs a peer-reviewed PRINTED thesis table (Kranenburg 2006 Table 5.2, 35/35 rows within tol 0.02 via live re-solve).
- **Adversarial refute:** verdict held; The audit's mixed/strict-verified verdict holds under adversarial scrutiny.

### `multi_echelon` — mixed

- **Anchor:** mixed; reproduces published: True.
- **Evidence:** SPLIT family; tier='mixed';
- **Caveat:** Per-subfamily tiers (audited_tier='mixed'): - serial = STRICT. One peer-reviewed printed cost (Snyder & Shen Ex6.1 47.65) re-run to 47.6654.
- **Adversarial refute:** verdict held; Audit holds (see detailed reason above).

### `dual_sourcing` — reference

- **Anchor:** published_action; reproduces published: True.
- **Evidence:** SOURCE IS PEER-REVIEWED: Gijsbrechts et al. (2022), MSOM, doi:10.1287/msom.2021.1064 (src/problems/dual_sourcing/literature/references.rs:83-88).
- **Caveat:** MANIFEST OVERSTATES: BENCHMARK_MANIFEST.json sets verification_tier="strict" for dual_sourcing, but under the strict definition (re-run a peer-reviewed printed ABSOLUTE COST from a table) this family is REFERENCE, not strict: it reproduces a published GAP PERCENTAGE (Figure-9 bar labels) measured against the repo's OWN bounded-DP optimum, and no published absolute cost exists f…
- **Adversarial refute:** verdict held; The audit is correct and I confirmed every load-bearing claim by reading the files and re-running the live reproduction.

### `joint_replenishment` — reference

- **Anchor:** published_action; reproduces published: True.
- **Evidence:** TIER = REFERENCE (reproduces a published ACTION, not a peer-reviewed printed cost). Evidence: 1.
- **Caveat:** Single subfamily (no split family / no 'mixed' needed). audited_tier=reference, NOT strict and NOT faithful, for these reasons: - NOT strict: the env does not re-run any number PRINTED in a peer-reviewed table.
- **Adversarial refute:** verdict held; Audit holds.

### `nonstationary_lot_sizing` — reference

- **Anchor:** companion_code_number; reproduces published: True.
- **Evidence:** VERDICT: reference (literature_verified=true, literature_verified_strict=false). The env genuinely re-runs and reproduces published rows, but those rows are AUTHOR COMPANION-CODE CSVs, not a peer-reviewed printed cost.
- **Caveat:** Per-subfamily: single subfamily only (Dehaybe 2024 lost-sales rolling-forecast lot-sizing) = REFERENCE. No multi-echelon-style split, so audited_tier is not 'mixed'.
- **Adversarial refute:** verdict held; The audit's reference verdict holds exactly and survives every adversarial demotion check.

### `ameliorating_inventory` — mixed

- **Anchor:** companion_code_number; reproduces published: True.
- **Evidence:** SPLIT FAMILY (mixed). (1) PERFECT-INFO LP BOUND = REFERENCE: the runner re-solves the LP and reproduces the published value to gap 5.54e-08.
- **Caveat:** Per-subfamily tiers: [perfect_information_LP_bound = REFERENCE: re-runs a Pahr-Grunow 2025 COMPANION-CODE upper_bound.json value (1991.9344293376805 spirits_0001; 2444.8010643781136 port_wine;
- **Adversarial refute:** verdict held; The audit holds.

### `decentralized_inventory_control` — mixed

- **Anchor:** closed_form_number; reproduces published: True.
- **Evidence:** HEADLINE TIER = reference (NOT strict). The 204 anchor is a CLOSED-FORM/COMPANION-CODE number, not a transcribed peer-reviewed printed cost, and it is reproduced ONLY by a simulator disconnected from the trainable env.
- **Caveat:** No multi_echelon/lost_sales subfamily split exists in this family; the meaningful split is closed_form_board_game vs env.rs.
- **Adversarial refute:** verdict held; see above

### `one_warehouse_multi_retailer` — faithful — NOT literature-verified

- **Anchor:** repo_native_style_no_anchor; reproduces published: False.
- **Evidence:** references.rs:582 VERIFICATION_PROBLEM_INSTANCE literature_verified=false ("Repo-native exact verifier", notes line 605); references.rs:97-106 KAYNOV_2024_REFERENCE (real paper IJPE 267, 109088, DOI 10.1016/j.ijpe.2023.109088);
- **Caveat:** Single family with 3 customer-behavior REGIMES (lost_sales / backorder / partial_backorder), NOT split into separately-tiered subfamilies, so audited_tier is a single 'faithful', not 'mixed'. Per-regime fidelity (repo's own measurements): lost_sales rows closest (~1-2.5%, the only regime where a row re-runs within tolerance via min_shortage), backorder ~-3.6 to -5.5%, partial_b…
- **Adversarial refute:** verdict held; The FAITHFUL / literature_verified=false verdict holds;

### `joint_pricing_inventory` — faithful — NOT literature-verified

- **Anchor:** closed_form_number; reproduces published: False.
- **Evidence:** VERDICT: FAITHFUL (literature_verified=false, NOT verified). All sources converge on faithful_unverified / no_published_number.
- **Caveat:** Single-instance, primary-only family (no subfamilies; audited_tier is a clean single 'faithful', not 'mixed').
- **Adversarial refute:** verdict held; The audit is correct;

### `procurement_removal_inventory` — faithful — NOT literature-verified

- **Anchor:** repo_native_style_no_anchor; reproduces published: False.
- **Evidence:** VERDICT: FAITHFUL (literature_verified=false). Single-family problem, no subfamilies — audited_tier='faithful', not 'mixed'.
- **Caveat:** Single family, no subfamilies (unlike multi_echelon/lost_sales), so tier is 'faithful' not 'mixed'. The env is structurally faithful to the Maggiar & Sadighian (2017) model (Theorem 3.4 interval-stock policy, Corollary 1 never-liquidate-returnable, Assumption 2 cost ordering c>s>=l verified in tests), and carries genuine real-paper citations (SSRN 3018984;
- **Adversarial refute:** verdict held; Verdict CONFIRMED as faithful, literature_verified=false.

### `random_yield_inventory` — faithful — NOT literature-verified

- **Anchor:** repo_native_style_no_anchor; reproduces published: False.
- **Evidence:** FAITHFUL / repo-native, NOT literature-verified. Single subfamily, so audited_tier=faithful (not mixed).
- **Caveat:** Single subfamily ("all-or-nothing batch yield"), so this is NOT a mixed/split family — audited_tier is uniformly faithful, no per-subfamily breakdown needed. The repo is scrupulously honest here and does NOT overclaim: it self-labels faithful_unverified / self-consistent-only / literature_verified=false consistently across references.rs, both READMEs, tests.rs, the manifest, an…
- **Adversarial refute:** verdict held; see above

### `vendor_managed_inventory` — faithful — NOT literature-verified

- **Anchor:** repo_native_style_no_anchor; reproduces published: False.
- **Evidence:** FAITHFUL — env is structurally faithful to Sui/Gosavi/Lin (2010) but reproduces NO published peer-reviewed number; all anchors are repo-native self-consistency or an instructor handout.
- **Caveat:** Single, non-split family (not multi_echelon-style), so audited_tier is a single tier "faithful", not "mixed". All 8 catalog instances are faithful/repo-native: PRIMARY_REFERENCE_INSTANCE (..._style_single_retailer), low/high_penalty + low/high_demand perturbations, VERIFICATION_PROBLEM_INSTANCE (repo-native exact-DP self-consistency verifier), the GOSAVI_CASE_STUDY handout news…
- **Adversarial refute:** verdict held; Verdict HOLDS at faithful / literature_verified=false / strict=false.

## Two tier observations (do not change the verified/excluded set)

The audit's strict/reference sub-labels differ from the manifest on two
families — both stay **verified**, only the sub-label moves, so the keep/exclude
line is unaffected:

- **`dual_sourcing`** — manifest `strict`, audit `reference`: the Gijs Figure-9
  anchor is a relative optimality **gap**, not a printed absolute cost the env
  re-runs.
- **`spare_parts_inventory`** — manifest `reference`, audit `strict`: Kranenburg
  (2006) Table 5.2 is a genuine peer-reviewed **printed table**, reproduced
  exactly by the analytical module (a different model from the trainable env).

These are recorded here for the record; the manifest `verification_tier` field
was left as-is pending a ledger reconciliation, since the boolean
literature-verified partition (the operative cut) is identical either way.
