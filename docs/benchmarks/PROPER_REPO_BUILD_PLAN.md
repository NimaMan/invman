# Proper Benchmark + Paper Repo — Build Plan

Derived from the 2026-06-06 cross-system audit of all 14 inventory families. Goal: turn a collection of faithful envs + scattered markdown logs into a **proper benchmark repo** where every headline number is (a) tied to a named instance, (b) reproducible by one command at a pinned seed protocol, and (c) honestly labeled verified / reference-verified / faithful-unverified.

Each work item carries: **scope** (one line) · **effort** S/M/L · **needs**: `CMA` (new optimizer-seed runs) / `code` (Rust or Python code, may need rebuild) / `docs` only.

Effort key: **S** ≈ <½ day · **M** ≈ 1–2 days · **L** ≈ ≥3 days.

---

## (a) Standard benchmark API per problem

> **Status (2026-06-12): the uniform EXECUTABLE surface now exists** as
> `invman/benchmarks/runners/`, covering **all 14 catalog families** (157
> reference instances) — `catalog.get(problem).load_instance(name)` returns a
> runnable `ReferenceInstance` (env params + published baselines +
> `run_baselines()` to re-run them on the live env + `compare()`). All 14 support
> load/baselines/compare; **lost_sales (+fixed), dual_sourcing, multi_echelon**
> additionally support `evaluate()` through the CMA-ES seam (the other 11 set
> `supports_evaluate=False` — their soft-tree rollout isn't in the
> `build_policy`/`get_model_fitness` seam yet, so `evaluate()` raises an
> actionable error). Worked reports: `scripts/benchmark_baselines/`. REMAINING:
> wire `evaluate` for the 11 metadata-only families (a `build_policy` +
> `rollout_fitness` branch each); the A1–A11 items below are the per-family
> Rust-accessor refinements (provenance split, extra bindings).

Target every problem family to expose the *same* Python surface so a benchmark consumer never has to parse Rust or markdown:

- `<problem>_list_reference_instances()` → names
- `<problem>_get_reference_instance(name)` → full param dict
- `<problem>_primary_reference_instance()` → the canonical instance
- `<problem>_exact_verification_instance()` (where an exact solver exists)
- a per-instance **`coverage_dimensions`** tag list (regime / demand / leadtime / cv / K / penalty …) — already collected in `BENCHMARK_MANIFEST.json`, fold into the structs
- a per-instance **`literature_verified`** boolean **plus a separate `reproduced_gap_pct`** field (provenance ≠ numerical reproduction — see OWMR)
- a `convention_caveat` string for cross-method comparisons (divergent gijs_2022-vs-VanRoy; padn optimal-proxy substitution; dual-sourcing CDI-as-proxy)

| Item | Scope | Effort | Needs |
|---|---|---|---|
| A1 | **OWMR**: split `literature_verified` (provenance) from new `reproduced_gap_pct`; add a single `<problem>_benchmark_card()` accessor (instance params + published rows + reproduced gaps + status in one call) | M | code |
| A2 | **ameliorating_inventory**: add `*_perfect_information_upper_bound` pyfunction (exposes `solve_upper_bound`) + `*_list/get_reference_instance` (family currently exposes NONE); add spirits_0002 / spirits_1002 to `REFERENCE_INSTANCES` | M | code |
| A3 | **multi_echelon/assembly**: add an assembly `*_exact_reduction_summary` binding (Rosling reduce + solve_from_local_costs) so the 22.759/52.536/27.530 costs are auditable without cargo | M | code |
| A4 | **multi_echelon/serial**: add a serial env-sim `*_summary` binding (the L0=1 self-consistency check) so env-sim reproduction isn't cargo-test-only | S | code |
| A5 | **decentralized_inventory_control**: expose `solve_optimal_policy` as a Python binding (currently Rust test-only); add instance-definition accessor returning data | M | code |
| A6 | **vendor_managed_inventory**: expose `solve_optimal_policy` (Rust-only today) so the benchmark reports a true optimality gap, not just learned-vs-heuristic; add reduced-slice instance registry | M | code |
| A7 | **procurement_removal_inventory**: add `*_removal_active_reference_instance` accessor (currently hardcoded twice — references.rs AND benchmark script can drift) | S | code |
| A8 | **random_yield_inventory**: parameterize the exact-DP binding beyond the single hardcoded VERIFICATION instance (allow L, p, cap sweeps) | M | code |
| A9 | **lost_sales**: add a per-grid-row `literature_verified` field to `reference_costs.rs` (today only `references.rs` anchor structs carry it) + relabel mislabeled `source='literature'` rows to `'computed'` (only L4-Poisson is true Zipkin) | M | code+docs |
| A10 | **nonstationary_lot_sizing**: add `*_list_reference_instances()` so the Python script stops duplicating the published rows by hand (drift risk vs references.rs) | S | code |
| A11 | **joint_replenishment / joint_pricing_inventory**: add per-instance accessors + a `feature_accessor` doc for jpi (build_raw_state is 2-dim but rollout uses 7-dim derived features) | M | code+docs |

---

## (b) Verification-debt list — snapshot_only → executing re-run

Each below currently asserts carried==published *literals* without executing the env, or was validated on a prior date but not re-run. Convert to executing assertions. (Full table in `VERIFICATION_LEDGER.md` Group-3/Debt section.)

| Item | Scope | Effort | Needs |
|---|---|---|---|
| V1 (HIGH) | **dual_sourcing l_r=3,4** (D4): wire an on-demand executing reproduction + record a dated artifact; today only l_r=2 has a fast executing test, l_r=3,4 are `#[ignore]`d | M | code (compute-heavy) |
| V2 (HIGH) | **ameliorating LP bound**: with A2's new binding, add a <2-min Python re-run asserting the bound within 1e-3 → upgrades ameliorating to verified_rerun | S | code |
| V3 (MED) | **divergent A3C rows** (D1): relabel as "published context, not reproduced" (repo has no A3C) — stop implying verification; OR implement an A3C comparator (L) | S (relabel) / L (implement) | docs / code+CMA |
| V4 (MED) | **gbk set2/set3** (D2): flag as NOT-reproduced (+223%) next to the verified set1/KT rows; recover the gated CEJOR transition spec if obtainable | S (flag) / L (recover) | docs / code |
| V5 (MED) | **van_oers 2024 Table 1** (D3): build the two-echelon serial-AM env to re-run, or drop from the card | L (build) / S (drop) | code / docs |
| V6 (LOW) | **dual_sourcing drift guards** (D5): clearly label `figure_9_gap_labels_are_frozen` as a drift guard, not the verification (the executing l_r=2 test is canonical) | S | docs |
| V7 (LOW) | **decentralized_inventory_control env.rs** (Group 4): either re-derive S'/supply-line bookkeeping to match the board-game iti1/iti2/wipi split (so env.rs reproduces 204), or adopt a published Clark-Scarf serial instance with a known optimum matching env.rs's order-after-demand convention | L | code |
| V8 (LOW) | **padn general-network**: recover Pirhooshyaran's exact OUL→position protocol (or compute correct local base-stock + a published reference cost); the env does not reproduce 47.65/72.04 under carried echelon levels | L | code |

---

## (c) Seed-robustness debt — every at_risk result → mean±std over ≥5 optimizer seeds

**Project mandate (MEMORY: seed-robust-reporting-standard):** report mean±std over ≥5 *optimizer* seeds, never single-seed/best-of-N. Almost every "beats heuristic" headline is currently a single CMA-ES seed (the large eval-SEM is demand-path CRN noise, NOT optimizer-seed variance). Build one shared `seed_robust_<problem>.py` pattern (model on the existing `seed_robust_mixed_distribution_assembly_network.py`) that loops ≥5 seeds and emits mean±std + per-seed range.

### HIGH (headline / paper-table claims that are single-seed or best-of-N)

| Item | Scope | Effort | Needs |
|---|---|---|---|
| S-H1 | **OWMR instance_13** (+6.44%) and **instance_14** + regime rows (3/9/10): paper-table wins are single-seed; run ≥5 seeds (instance_12 already has a robust +4.99% — reconcile paper to it) | M | CMA |
| S-H2 | **multi_echelon/divergent settings 1&2** (-14.4% "beats A3C"): currently best-of-N single optimizer seed → ≥5-seed mean±std before "beats A3C" survives | L | CMA (slow env) |
| S-H3 | **multi_echelon/gbk set1 (22.4/26.7%) + KT (~37%)**: only N=2 seeds (123,777) despite large margins → add ≥3 more seeds | M | CMA |
| S-H4 | **lost_sales vanilla 22/24-win + fixed-cost surface**: every learned cell is a single optimizer seed; sub-percent "beats myopic2 -1.20%" needs ≥5 seeds | L (whole surface) | CMA |
| S-H5 | **joint_replenishment 6/16 MOQ-beats** (setting 5 +13.05% etc.): single-seed; the setting-10 flip is best-of-N=2 inside noise → ≥5 seeds | M | CMA |
| S-H6 | **ameliorating_inventory** spirits_0001/port_wine/spirits_1002 (+450/+278/+524%): huge margins but single-seed → ≥5 seeds to make robust | S (fast env) | CMA |

### MED

| Item | Scope | Effort | Needs |
|---|---|---|---|
| S-M1 | **perishable_inventory** 5 "beats gate" rows (+0.70..+2.21%): single-seed; MEMORY already flags "perishable x5" as a MED debt → `seed_robust_perishable_inventory.py` | M | CMA |
| S-M2 | **random_yield_inventory**: sweep is 4 seeds (one short of ≥5); add a 5th, make the d1-linear-b8 number canonical (the d3 headline contradicts saved evidence) | S | CMA |
| S-M3 | **joint_pricing_inventory** (+25.15%): single training seed; no seed-sweep runner exists | M | CMA |
| S-M4 | **nonstationary_lot_sizing** (beats DP 8/8): single seed=1234, best-of-population; no multi-seed runner | M | CMA |
| S-M5 | **dual_sourcing** 2 "BEATS CDI" rows + factor-screen: single-seed; paper already downgrades to "matches" — reconcile RESULTS_SUMMARY.md / MEMORY to the paper | S | CMA + docs |

### LOW (no beat claimed, but seed-robust reporting still missing)

| Item | Scope | Effort | Needs |
|---|---|---|---|
| S-L1 | **vendor_managed_inventory** low_penalty "flip to win" (-0.31% from seed777, -0.06% at default): two-seed best-of → ≥5 seeds; robust verdict is a marginal tie | S | CMA |
| S-L2 | **spare_parts_inventory** (+1.34% vs best-constant, 6.84% on a different block): single seed; margin << cost std → ≥5 seeds | S | CMA |
| S-L3 | **procurement_removal_inventory**: soft_tree never beats heuristic anyway, but add ≥5-seed reporting for completeness | S | CMA |

---

## (d) Per-problem README "card" gaps

Every family needs a top-level README benchmark card: instance list, which cells are literature vs reference-impl vs computed, the single-seed/optimal-proxy caveats, and the exact reproduce command + expected number + tolerance. Currently scattered across 5+ READMEs / dated markdown logs per family.

| Item | Scope | Effort | Needs |
|---|---|---|---|
| C1 | One **`BENCHMARK_CARD.md` template** + generator that reads `BENCHMARK_MANIFEST.json` and emits a per-problem card (avoids hand-drift) | M | code+docs |
| C2 | **lost_sales / dual_sourcing / OWMR / multi_echelon**: cards stating literature-vs-computed per row, the optimal-proxy/convention caveats, expected values + tolerances | M | docs |
| C3 | **ameliorating / vmi**: card clarifying which env each Python binding targets (faithful vs legacy reduced model) | S | docs |
| C4 | **faithful_unverified families** (jpi / procurement / random_yield / vmi / spare_parts-trainable): card carrying the explicit `faithful_unverified` status so consumers don't misread self-consistency anchors as literature | S | docs |

---

## (e) Reproducibility spine

Today many headline numbers live only in gitignored `outputs/` + markdown; trained parameter vectors aren't committed, so learned rows can't be re-evaluated without retraining.

| Item | Scope | Effort | Needs |
|---|---|---|---|
| R1 (HIGH) | **Commit a results table tied to seeds**: per headline number store {artifact path, exact command, optimizer seed(s), eval seeds, expected value, tolerance}. Persist the trained parameter vector (or a small checkpoint) for each paper-table row | L | code+docs |
| R2 (HIGH) | **One script to regenerate ALL paper tables** from committed artifacts + manifest (e.g. `scripts/regenerate_paper_tables.py`) | M | code |
| R3 (MED) | **Pin a seed protocol**: optimizer seeds {a set of ≥5}, eval CRN seeds, warm-up ratio — documented once and referenced by every runner | S | docs |
| R4 (MED) | De-gitignore (or mirror to a tracked results dir) the specific JSONs backing paper numbers: setting5_vi_optimum_gap.json (jrp), owmr campaign logs, ameliorating RESULTS_FULL_BUDGET.md numbers | M | code+docs |
| R5 (LOW) | Commit the **independent Python DP cross-check scripts** whose results are quoted but not checked in (procurement_removal verification/README; joint_pricing) | S | code |

---

## (f) Cleanup

| Item | Scope | Effort | Needs |
|---|---|---|---|
| F1 | ~~Remove dead scratch scripts importing the deleted `invman.policies.soft_tree`~~ **RESOLVED — claim was stale/false (audited 2026-06-11): these 7 scripts are LIVE, not bricked.** `scripts/perishable_inventory/{run_paper_benchmark.py,common.py}`, `scripts/procurement_removal_inventory/{common.py,train_soft_tree_reference.py,validate_against_exact_dp.py}`, `scripts/random_yield_inventory/{common.py,train_soft_tree_reference.py}` were MIGRATED off `invman.policies.soft_tree` (the only references to that path are now docstrings describing the migration). All 7 parse + `py_compile` clean; entry-points load via `--help`; each `common.py` imports live modules (`invman.policy.Policy`, `invman_rust`) and is depended on by live sibling runners (perishable: `run_practical_benchmark.py`, `validate_against_papers.py`, `train_soft_tree_reference.py`; procurement_removal + random_yield: their `train_soft_tree_reference.py`/`validate_against_exact_dp.py`/`seed_robust_*`/`summarize_*`). `invman/policies/` and `invman/problems/` confirmed absent, but no live script imports them. **Nothing removed.** (Matches `docs/benchmarks/CLEANUP_2026_06_06.md`, which already had these as LIVE.) | — | done |
| F2 | **`src/case_studies/hormuz_strait/`**: stray case-study tree (data/timelines/scenarios/maritime_traffic/sources) unrelated to the 14 benchmark families. Decide: move to a separate repo or document as an explicit out-of-benchmark case study; do not let it pollute the benchmark surface | S | docs (decide) / M (extract) |
| F3 | **Prune locked worktrees**: 8 `.claude/worktrees/*` exist (wf_* and agent-*); per MEMORY (invman-env-rewrite-worktrees) the procurement_removal faithful env is already cherry-picked (f9b6814) and the ameliorating rewrite worktree is incomplete. Audit each, cherry-pick anything live, then `git worktree remove` | M | code |
| F4 | **Rename stale identifiers**: vmi Rust symbols still carry `GIANNOCCARO_2010_*` though re-attributed to Sui/Gosavi/Lin (needs Rust rebuild) | S | code |
| F5 | **Consolidate dated markdown logs** (PPO_BEAT_CAMPAIGN_*, MIXED_ASSEMBLY_SEED_ROBUST_*, RESULTS_*.md) into the manifest-driven card system from C1 | M | docs |

---

## Suggested execution order (highest leverage first)

1. **A1+A2 (accessors) + C1 (card generator)** — unlocks consistent cards and the ameliorating bound; cheap, enabling.
2. **V2 + V1** — upgrade ameliorating to verified_rerun; close the dual_sourcing l_r=3,4 re-run debt.
3. **S-H1, S-H6, S-M2** (fast envs first: OWMR, ameliorating, random_yield) — convert the at_risk headline beats to ≥5-seed mean±std; reconcile paper tables.
4. **R1+R2 (reproducibility spine)** — commit artifacts + one paper-table regenerator.
5. **F3 (cleanup)** — prune worktrees. (F1 is resolved: the "bricked scripts" were a stale/false claim — those 7 scripts are live; nothing to delete.)
