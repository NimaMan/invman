# Plan

Last checked: 2026-06-04.

Branch: `network-inventory-serial-clark-scarf-verification`

HEAD: `7fceebf` (serial + ameliorating autoresearch bindings & results committed)

Working tree: still dirty with pre-existing post-migration churn (paper edits, CPU-limit
scripts, docs). The serial + ameliorating Track-2 work is now committed; the remaining
uncommitted paths are Tier-2 infrastructure, not Tier-1 problem results.

## Objective

Make `invman` a reproducible Rust-backed benchmark and paper workspace for compact CMA-ES inventory-control policies, with honest comparisons against exact solvers, repo heuristics, and published literature baselines.

The near-term objective is to stabilize the post Python-cleanup migration so the paper claims, benchmark catalog, tests, and runnable scripts all point at the same current source of truth:

- problem dynamics and benchmark references mostly route through `invman_rust`
- Python policy/search code lives in the flattened `invman/` package
- `literature_verified` is reserved for repo exact or heuristic implementations that actually reproduce a published number
- learned-policy rows are reported as learned-policy comparisons, not as literature verification

## Tier 1 — Per problem: train > heuristic → build the experiment instance → add to the paper

This is the headline workstream. For **every literature-verified problem** run one pipeline,
in this order, and track each problem against it:

1. **Train a policy that beats the literature heuristic.** A compact CMA-ES policy must beat
   the problem's literature heuristic on the literature instance, evaluated **like-for-like**
   (same Rust env, same paired common-random-number protocol, same horizon/seeds). Where the
   comparator is a **proven optimum** rather than a heuristic, the honest ceiling is to
   **match** it — never claim a "beat".
2. **Build the literature experiment instance.** Lock the published parameter set as THE
   experiment instance, with its strongest in-repo heuristic as the keep/discard gate. Most
   instances are already coded in `references.rs`; the work is choosing the gate and the
   eval protocol, not re-deriving dynamics.
3. **Add it to the paper.** If the learned policy clears the bar (beats the heuristic, or
   matches the optimum), write it into `paper/learning_inventory_control_policies_es.tex` as a
   *result* — promoting from the appendix env-fidelity table where it currently sits as a
   validation-only row.

**Honest-comparator rule (carries the verification bar forward).** `literature_verified` means
a repo test re-runs the env and reproduces a *published number*. Learned-policy rows are
learned-vs-comparator comparisons, **not** literature verification. Every problem is one of:
- **`heuristic_to_beat`** — beating the comparator is a legitimate win.
- **`true_optimum_match_only`** — the comparator is a proven optimum; matching it is the
  ceiling, and any "beat" is an env/eval bug, not a result.
- **`bound_gap`** — the comparator is an upper/lower bound; report the gap, and only claim a
  "beat" against the in-repo heuristic gate.

### Tier-1 status (the 10 literature-verified problems)

| # | Problem | Comparator | Current learned result | Instance | In paper | Stage / next |
|---|---|---|---|---|---|---|
| 1 | lost_sales_vanilla | heuristic_to_beat | **beats** Myopic-2 (4.749 vs 4.817, −1.4%); 22/24 instances | ✓ | ✓ result | **CLOSED** |
| 2 | lost_sales_fixed_order_cost | heuristic_to_beat | **beats** mod-(s,S,q) (8.733 vs 9.174, −4.8%); 47/48 | ✓ | ✓ result | **CLOSED** |
| 3 | dual_sourcing | true_optimum_match_only | **matches** CDI on 6/6, beats on 2; beats A3C everywhere | ✓ | ✓ result | **CLOSED** |
| 4 | perishable_inventory | heuristic_to_beat | **beats** base-stock gate (FIFO +1.16%, LIFO +0.82%); ≈VI-opt | ✓ | ✗ | **WRITE-UP** |
| 5 | general_backorder_fixed_cost | heuristic_to_beat | **beats** Geevers benchmark −22%…−27%; below PPO 8,714 | ✓ | appendix only | **WRITE-UP** |
| 6 | multi_echelon_serial | true_optimum_match_only | **matches** Clark-Scarf 47.65 (47.6554, 99.99%) | ✓ | appendix only | **WRITE-UP** |
| 7 | joint_replenishment | heuristic_to_beat | **beats** MOQ −13.8% (setting 5); +3.14% over VI-opt | ✓ | appendix only | **WRITE-UP** (regen artifacts) |
| 8 | one_warehouse_multi_retailer | heuristic_to_beat | **ties** tuned base-stock (0.0%) on symmetric K=3 | ✓ | ✗ | **TRAIN** (asymmetric) |
| 9 | ameliorating_inventory | bound_gap | **beats** order-up-to gate +79.8%; 95% below LP bound | ✓ | ✗ | **DECIDE** (action geometry) |
| 10 | production_assembly_distribution_network | heuristic_to_beat (research) | **beats** own pairwise base-stock −4.96% | ✓ | appendix only | **DECIDE** (`literature_verified=false`) |

### Stage CLOSED — already written into the paper as a result (no Tier-1 action)

#### 1. lost_sales_vanilla
- **Comparator:** Myopic-2 (Zipkin 2008 Table 3a) = 4.82; DP optimum 4.73 is a floor.
- **Result:** Tree-1 = 4.749 beats Myopic-2 by ~1.4%, within 0.4% of the DP optimum; learned
  beats Myopic-2 on **22/24** instances. Paper Table `tab:results-vanilla-lost-sales`.
- **Open (optional):** two p=19 MMPP2+ deep-pipeline (L=8,10) cells still lost to SVBS/Myopic-2
  → a regime-aware / longer-pipeline policy is the only remaining lever. Not blocking.

#### 2. lost_sales_fixed_order_cost
- **Comparator:** modified (s,S,q) (Bijvank 2015 family) = 9.174 on canonical L4/p4/K5;
  separate proven optimum 11.46 on the Bijvank L2/p14/K5 validation instance (match-only there).
- **Result:** NN gated-ordinal = 8.733 beats mod-(s,S,q) by ~4.8%; **47/48** instances
  instance-best. Paper `tab:results-fixed-cost-lost-sales` + validation `tab:fixed-cost-validation`.
- **Cleanup before publication:** (a) NN-categorical row duplicates the linear-categorical
  numbers (daggered, re-verify); (b) excluded divergent CMA runs documented; (c) `invman.problems.*`
  import gap means the legacy re-run scripts fail — Tier-2 fix.

#### 3. dual_sourcing
- **Comparator:** capped dual-index (CDI), a ≤0.11% proxy for the bounded-DP optimum →
  `true_optimum_match_only`. Prior DRL A3C (0.51–1.85% gap) is the beatable published baseline.
- **Result:** learned soft tree **matches** CDI on all 6 Gijsbrechts Fig-9 rows, is
  statistically below it on 2, and clears A3C on every row. Paper `sec:dual-sourcing` /
  `tab:ds-results`.
- **Reproducibility debt:** the canonical 70-seed report (`outputs/dual_sourcing_policy_search/
  final_report.json`) is git-untracked; archive it so the paper table regenerates from a tracked
  source. Reconcile the 3-seed broad-grid run (+0.18–0.55%) as explicitly *not* the reported run.

### Stage WRITE-UP — result clears the bar; only the paper section is missing

#### 4. perishable_inventory  ← cleanest next paper add
- **Comparator:** best base-stock gate, re-scored on the same MC estimator: −1475.08 (FIFO) /
  −1565.98 (LIFO). De Moor 2022 / Farrington 2025 VI optimum −1457 / −1553 is context.
- **Result (committed 222d46f):** depth-2 oblique linear soft tree, validation-block selection
  — FIFO −1457.90 (**+1.16%** over gate, 9.5× SEM, ≈VI-opt), LIFO −1553.16 (**+0.82%**, 6.6× SEM).
- **Instance:** `de_moor2022_m2_exp2_l1_cp7_fifo` (+ LIFO sibling) — the only two with an in-crate
  exact-MDP verifier.
- **Next:** write a perishable results subsection (cite De Moor 2022 + Farrington 2025). Report
  the **base-stock-gate beat** as the like-for-like win; frame VI-optimum proximity as context
  only (estimator-mismatch caveat). Selection MUST stay on the disjoint validation block (eval-block
  selection flips it to a −0.49% loss).

#### 5. general_backorder_fixed_cost
- **Comparator:** Geevers constant node-base-stock benchmark = 10,467 (repo reproduces 10,355,
  −1.1%). Published PPO best 8,714 is a cross-protocol reference, not the gate.
- **Result (committed 7922afd):** learned node-base-stock-targets soft tree = 8,035 (**−22.4%**
  vs benchmark; seed 777 → 7,591, −26.7%), below the published PPO. gen-0 warm-start reproduces
  the benchmark, so the win is the CMA delta.
- **Instance:** `geevers2023_general_set1` (the ONLY verified row; sets 2/3 are
  `literature_verified=false`, gated-journal transition spec — exclude). Note `PRIMARY_REFERENCE_INSTANCE`
  misleadingly points at set 3; train/verify on set 1.
- **Next:** promote from the `tab:additional-env-validation` appendix row to a reported result.
  Flag honestly: env name says "fixed_cost" but charges holding+backorder only; PPO comparison is
  cross-protocol (suggestive, not head-to-head).

#### 6. multi_echelon_serial  (match-only)
- **Comparator:** Clark-Scarf / Snyder & Shen Ex 6.1 **proven optimum 47.65** → match-only.
- **Result (committed 7fceebf):** warm-started direct-level soft tree = **47.6554 (99.99% match,
  +0.011%)** — ties the optimum within the env's +0.06% reproduction band. Beating is impossible.
- **Instance:** Snyder & Shen Example 6.1 (downstream stage L=1, the only faithful regime; env
  under-counts when downstream L≥2).
- **Next:** if promoting from the appendix fidelity row, frame strictly as *"reproduces the proven
  Clark-Scarf optimum to within +0.06%"*, never "beats". Also: the dir `README.md` is **stale** —
  it still says the Python binding is missing and the comparison is BLOCKED; update it (binding now
  exists and ran).

#### 7. joint_replenishment
- **Comparator:** MOQ heuristic = 7,593.66 (setting 5); VI optimum 6,347.11 is the floor
  (Vanvuchelen Fig-3 action anchor is the literature verification).
- **Result (committed e08c326):** depth-3 oblique soft tree = 6,546.18 (**−13.79%** vs MOQ, cheaper
  on all 4096/4096 paths; +3.14% over VI-opt, closing **84%** of MOQ's gap).
- **Instance:** `vanvuchelen2020_small_scale_setting_5` (the only setting with a published true
  optimum). Multi-setting is **not** paper-ready (6/16 wins; h=5/b=95 cluster loses) — single
  setting only, unless the base-stock-anchored action adapter is built.
- **Next (blocker):** the trained model + `setting5_vi_optimum_gap.json` live only in a worktree,
  not in `outputs/` — **regenerate in the main tree** before write-up. Then add a result row
  (honest framing: "closes 84% of MOQ's gap to the paper's own VI optimum").

### Stage TRAIN — needs new training before it can enter the paper

#### 8. one_warehouse_multi_retailer
- **Comparator:** Kaynov 2024 tuned echelon base-stock + allocation gate (Table A.3). Published
  PPO beats base-stock by 12–22% on the partial-backorder instances.
- **Result:** symmetric K=3 instances (1/6/7/11) all **tie** at 0.0% — CMA from warm-start finds
  no profitable deviation because symmetric Poisson(3) base-stock is provably near-optimal. No win
  is possible there.
- **Plan:** pivot to the **asymmetric / high-CV partial-backorder instances** `kaynov2024_instance_12/13/14`
  (Kaynov's own PPO beats base-stock 20–21% there → real exploitable structure). Switch the action
  design from `symmetric_echelon_targets` to **`direct_orders` / `vector_quantity`** (per-retailer
  orders) — the symmetric geometry cannot express asymmetric policies.
- **Next:** run `autoresearch_one_warehouse_multi_retailer.py --budget full` on instances 12/13/14,
  `{constant,linear}` leaves, `--warm_start_at_best_base_stock`, allocation `{proportional,min_shortage}`,
  CPU-capped. Require a held-out flip beyond SEM to claim a win; otherwise report the honest
  matched-and-dominated framing (learned ties a tuned heuristic that already ≤ published PPO).

### Stage DECIDE — result exists but a framing/scope decision blocks paper inclusion

#### 9. ameliorating_inventory  (bound_gap)
- **Comparator:** perfect-information LP **upper bound** 1991.93 (spirits_0001) / 2444.80
  (port_wine) — a bound, not a heuristic. In-repo tuned order-up-to gate (≈20.8) is the beatable comparator.
- **Result (committed 7fceebf):** price-reactive single-purchase soft tree = 100.54, **+79.8%**
  over the order-up-to gate, but **94.95% below the LP bound** — NOT comparable to the paper's
  ~3.5% DRL gap (the bound assumes full 3-part LP issuance; our policy controls only scalar purchase).
- **Decision required:** either (a) **widen the action geometry** to the full 3-part action (add
  production-target heads) to chase the ~3.5% gap, or (b) **scope the claim** to purchase-only and
  publish the honest "beats order-up-to, gap-to-bound is loose" story.
- **Next:** run `--budget full` on spirits_0001 + port_wine for committed numbers; add a Pahr &
  Grunow 2025 entry to `references.bib` and a paper section **only after** the geometry/scope
  decision (the current single-purchase gap is not paper-grade).

#### 10. production_assembly_distribution_network
- **Comparator:** the case3 gate is the env's **own** best pairwise base-stock (60.24) — a research
  baseline, NOT a published optimum. The env is **`literature_verified=false`** (only the single-node
  newsvendor 127.11 row is verified; the serial 47.65 is structurally unreachable here — its home is
  problem #6).
- **Result (committed f4f3dc3):** learned soft tree = 57.25, **−4.96%** vs the own-heuristic gate
  (robust across seeds/depths).
- **Decision required:** either (a) present honestly as a *research result on a faithful-but-not-
  literature-anchored env* (learned vs env's own best base-stock), or (b) first make the env
  literature-verified by recovering Pirhooshyaran's exact OUL→local-position protocol so it
  reproduces a published cost, then re-baseline. Path (a) is shippable now; path (b) is open work.
- **Next:** make the framing decision; do **not** dress this as "beats a literature benchmark".

### Not yet eligible for Tier 1 (need a literature anchor / faithful env first)

- **joint_pricing_inventory** — no published worked example wired (need Federgruen & Heching /
  Petruzzi & Dada). A `train_soft_tree_reference.py` stub exists but there is no verified anchor.
- **random_yield_inventory** — needs a per-instance published number (Yan 2026 / Chen 2018).
- **procurement_removal_inventory** — faithful Maggiar & Sadighian env exists on worktree branch
  (`f9b6814`), honestly `literature_verified=false` (NPV is graphical-only). Cherry-pick is a
  pending user greenlight; even then it stays a structure-anchored, not number-anchored, problem.
- **Honest-`false` families** (decentralized_inventory_control, multi_echelon/{assembly, divergent},
  nonstationary_lot_sizing, spare_parts_inventory, vendor_managed_inventory) — documented as
  non-reproducible; not Tier-1 candidates until an anchor is found.

---

## Tier 2 — Enabling infrastructure & repo hygiene

The sections below predate the tiering. They are the **Tier-2 substrate** the Tier-1 re-runs
depend on (build path, import migration, CPU caps, source-of-truth freeze). Resolve the
highest-priority blocker (post-migration import fallout) before any Tier-1 *re-run*; the existing
Tier-1 *committed results* do not depend on it.

## Current State

- The project has moved beyond the original lost-sales-only scope into a cross-problem manuscript and benchmark repo covering lost sales, fixed-cost lost sales, dual sourcing, multi-echelon variants, and additional exploratory families.
- The active paper file is `paper/learning_inventory_control_policies_es.tex`; the old fixed-cost standalone paper file is deleted in the working tree and a new `paper/invman_lostsales.tex` is untracked.
- The Python package is currently flattened (`invman/policy.py`, `invman/policy_registry.py`, `invman/rollout_fitness.py`, etc.). There is no `invman/problems/` package and no `invman/policies/` package in the current tree.
- New local work includes CPU concurrency guards (`invman/cpu_limits.py`), capped child-process worker environments, sampled ES population metadata tests, a continuous soft-tree Rust action head, serial Clark-Scarf bindings/runner, and faithful average-profit ameliorating-inventory bindings/runner.
- The installed `invman_rust` module in the active Python environment already exposes the newly added serial multi-echelon and ameliorating-inventory average-profit bindings.
- `python numerical_experiments/run.py --list` works and lists the current ready/exploratory suite catalog, but the catalog should not be treated as proof that every listed script is runnable under the current API.

## Verification Snapshot

Commands that currently pass:

- `python numerical_experiments/run.py --list`
- `python -m pytest tests/test_cpu_limits.py -q` (`5 passed`)
- help/import checks for:
  - `scripts/lost_sales/benchmark_full_suite.py`
  - `scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`
  - `scripts/dual_sourcing/benchmark_full_suite.py`
  - `scripts/one_warehouse_multi_retailer/run_paper_benchmark.py`
  - `scripts/multi_echelon/autoresearch_multi_echelon.py`
  - `scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py`
  - `scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py`

Commands that currently fail:

- `python -m pytest tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py tests/test_numerical_experiments_catalog.py tests/test_cpu_limits.py -q`
  - fails during collection because tests import missing `invman.problems.*`
- `python -m pytest tests/test_soft_tree_policy.py tests/test_multi_echelon_problem.py -q`
  - fails during collection because `tests/test_soft_tree_policy.py` imports missing `invman.policies`
- `python scripts/lost_sales/autoresearch_lost_sales.py --help`
  - fails because it imports missing `invman.policies.registry`
- `cargo test --manifest-path rust/Cargo.toml -q`
  - fails at link time with unresolved Python symbols from PyO3; the supported build route appears to be `python scripts/build_rust_extension.py` / maturin with `PYO3_PYTHON`

## Latest Limiting Factors

1. Post-migration import fallout is the highest-priority blocker. Many tests, docs, and older scripts still reference deleted paths such as `invman.problems.*`, `invman.policies.*`, and `invman.policies.soft_tree`. Some newer scripts are self-contained and Rust-routed, but the test suite is not yet a reliable health signal.

2. The repo needs one compatibility decision: either restore thin compatibility shims for the old Python package paths, or migrate every remaining caller to the flattened Python API plus `invman_rust`. Mixing both without an explicit boundary will keep producing false-ready scripts.

3. The Rust verification path is not clean as raw Cargo. `cargo test` does not link because the crate is a PyO3 extension module. Either document the maturin/Python test path as canonical or split Rust unit testing so verification can run without unresolved Python symbols.

4. Installed-extension/source drift is possible. The active environment exposes the new bindings, but the Rust source is dirty. After Rust edits, rebuild with `python scripts/build_rust_extension.py` and record which source commit/artifact produced benchmark numbers.

5. Benchmark readiness is inconsistent. The catalog lists ready suites, while recommended sanity tests and some legacy runners fail. "Ready" should mean runnable under the current API, not just present in `numerical_experiments/catalog.py`.

6. CPU oversubscription was recently patched and the new CPU-limit tests pass, but every long-running benchmark launcher and subprocess path still needs an audit for `normalize_args_cpu_limits(...)` and `cpu_limited_environ(...)`.

7. Reporting discipline is now a hard constraint. `literature_verified` should only be set when repo exact/heuristic code reproduces a published number. Published DRL/A3C/PPO rows are comparison rows, not repo-verified algorithms.

8. Dual sourcing is still mainly policy-geometry limited. Current evidence favors factorized capped-delta / capped-dual-index coordinates, with row-conditioned geometry: axis-linear for `l_r = 2`, tighter axis-constant small-cap trees for `l_r in {3,4}`. `autoresearch/dual_sourcing_policy_search/run_factor_screen.py` is still documented as carrying old broken imports.

9. Serial Clark-Scarf is match-only. The comparator is a true optimum, so the learned-policy result can tie the optimum within simulation error but should never be framed as beating it.

10. Faithful ameliorating inventory is bound-limited. The current single-purchase learned policy can beat the simple order-up-to heuristic, but it remains far below the perfect-information LP upper bound and is not comparable to the paper's full three-part-action DRL gap.

11. Several exploratory families still carry honest blockers: missing published anchors, self-consistency-only exact checks, stale old imports, or learned policies that still lose to tuned heuristics. Do not promote them into headline claims until their runners and verification status are clean.

12. The paper workspace is in heavy churn. There are deleted docs, a deleted old paper file, a new untracked paper file, regenerated figures/PDFs, and updated manuscript claims. Freeze the source of truth before doing final benchmark or Overleaf work.

## Rust-First Migration

Target direction: make the Rust crate the main project surface instead of a nested `rust/` folder. The repo root should eventually look like the Rust project root, with Python kept as a tier-two support layer for bindings, experiments, papers, and orchestration.

Proposed target layout:

- `Cargo.toml`, `Cargo.lock`, `src/`: primary Rust crate at repo root
- `src/problems/`, `src/core/`, `src/case_studies/`: tier-one Rust source domains
- `python/invman/` or `bindings/python/invman/`: tier-two Python API, CMA-ES runners, and compatibility shims
- `scripts/`, `numerical_experiments/`, `paper/`, `docs/`, `autoresearch/`: tier-two project support surfaces
- old `rust/`: removed after redirects/docs/build scripts have been updated

Migration steps:

1. Move `rust/Cargo.toml`, `rust/Cargo.lock`, `rust/pyproject.toml`, and `rust/src/` to the repo root equivalents.
2. Update `scripts/build_rust_extension.py`, setup/build docs, and any `--manifest-path rust/Cargo.toml` references to use root `Cargo.toml`.
3. Update paths in tests, scripts, docs, paper notes, and README files from `rust/src/...` to `src/...`.
4. Decide where the Python package lives after the move: either keep `invman/` at the root as a support package or move it under `python/invman/` with packaging metadata adjusted.
5. Rebuild `invman_rust` through the canonical maturin path and confirm the installed extension exposes the same bindings.
6. Rerun the minimum verification gate: catalog listing, CPU tests, focused Rust-binding Python tests, and import checks for ready benchmark scripts.
7. Remove the empty/obsolete `rust/` folder only after all docs and scripts no longer depend on it.

Migration risks:

- Path churn can invalidate benchmark scripts, docs, paper references, and Overleaf push assumptions.
- Python editable-install metadata may need a new package layout if `invman/` moves under `python/`.
- Cargo/PyO3 testing remains unresolved unless the migration also defines the canonical Rust verification command.
- Existing untracked/generated artifacts should not be moved blindly; separate source moves from output cleanup.

## Next Work

1. Use `rg "invman\\.problems|invman\\.policies"` to inventory every stale import and choose the shim-vs-migration strategy.
2. Fix the minimum verification gate first: CPU tests, catalog listing, core benchmark script imports, and focused pytest files that should still apply after the migration.
3. Rebuild `invman_rust` with `python scripts/build_rust_extension.py` after the Rust source changes, then rerun focused Python checks that exercise the new bindings.
4. Update `numerical_experiments/catalog.py`, README files, and paper notes only after runnable state and verification status match.
