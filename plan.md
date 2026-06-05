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
- The `scripts/lost_sales/`, `scripts/lost_sales_fixed_order_cost/`, and stable dual-sourcing lanes no longer contain deleted top-level `invman.problems.*` or `invman.policies.*` imports.
- The lightweight fixed-cost/lost-sales helper scripts have been migrated to the flattened Python API and Rust backend defaults: `autoresearch/fixed_cost_ordinal_stability/replay_exact_ordinal.py`, `scripts/raw_state_single_policy_probe.py`, and `scripts/evaluate_saved_policy.py`.
- `autoresearch/fixed_cost_ordinal_stability/ablate_state_drift.py` is now explicitly archived because it depended on deleted Python env/model APIs and an old torch checkpoint format.
- The perishable, joint-pricing, random-yield, and procurement-removal common helpers now build current `invman.policy.Policy` soft-tree descriptors instead of importing the removed `invman.policies.soft_tree.SoftTreePolicy`.
- README/agent/autoresearch active-surface docs now point at root Rust sources, flattened `invman/` support modules, and current benchmark helper scripts instead of deleted `invman.problems.*` / `invman.policies.*` packages.
- Fixed-cost heuristic benchmark summaries now use `invman_rust.lost_sales_fixed_heuristics_all_detailed`, which exposes winning `(s,S)`, `(s,nQ)`, and modified `(s,S,q)` params plus top candidates while preserving the legacy cost-only binding.
- Fixed-cost fixed-policy rollouts now also expose `invman_rust.lost_sales_fixed_policy_trace_from_demands`, returning per-period inventory position, pipeline, order quantity, arrivals, ending inventory, cost, and warm-up activity flags for supplied demand paths. The fixed-cost diagnostic script can now report a bounded deterministic trace sample for the searched modified `(s,S,q)` heuristic.
- Current `Policy` descriptors now have Rust action-evaluation bindings for soft-tree, linear, and NN backbones. The fixed-cost diagnostic's coarse learned-policy state-grid action histogram uses these bindings instead of the removed Python policy/env internals.
- Learned lost-sales policies now also have Rust supplied-demand trace bindings for soft-tree, linear, and NN backbones. The fixed-cost diagnostic can report deterministic rounded-mean demand trace samples for saved learned policies as well as for the searched modified `(s,S,q)` heuristic.
- The fixed-cost diagnostic can now archive the exact learned-policy trace, heuristic trace, and coarse action-summary JSON via `--output_json`, so these artifacts can be reviewed for paper/appendix inclusion without changing the benchmark regeneration path.
- Cargo now separates Rust-native verification from Python extension builds: default features are Rust-native, `python scripts/build_rust_extension.py` enables the `python-extension` feature for maturin/PyO3, and plain `cargo test --manifest-path Cargo.toml -q` links and passes.
- `numerical_experiments/run.py` now propagates the shared CPU/thread cap to child suite scripts, and the ready OWMR paper benchmark writes reports under root `src/problems/...` instead of recreating the removed `rust/src/...` tree.
- The exploratory-runner CPU audit has been extended from ad hoc `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` setup to the shared `invman.cpu_limits` helper: ameliorating average-profit, general backorder fixed-cost, serial multi-echelon, perishable, joint-replenishment learned/evaluator/autoresearch, OWMR autoresearch/learned/asymmetric runners, vendor-managed inventory, production-assembly, and the lost-sales paper-suite wrapper.
- Root-Rust layout assumptions now have executable regression coverage in `tests/test_rust_first_migration.py`: root `Cargo.toml`/`src/` are treated as source of truth, old `rust/Cargo.toml`/`rust/src` are forbidden, active source surfaces plus the tier-two `invman/` package and root build files are scanned for nested-Rust paths, and active scripts/docs/tests are scanned for live imports from deleted `invman.problems.*` / `invman.policies.*`.
- The perishable and nonstationary practical benchmark helpers no longer default to deleted `rust/src/...` artifact paths; they read/write under root `src/problems/...`. Their shared `invman.benchmarks.practical` report helper has been restored as a tier-two Python utility for JSON dataset loading and JSON/Markdown report writing.

## Verification Snapshot

Commands that currently pass:

- `python scripts/build_rust_extension.py`
- `python setup.py --name` (`invman`; latest run 2026-06-04)
- `cargo check --manifest-path Cargo.toml -q` (latest run 2026-06-04)
- `cargo test --manifest-path Cargo.toml -q` (`167 passed`, `1 ignored`; latest run 2026-06-04, 67.67s)
- `python numerical_experiments/run.py --list`
- `python numerical_experiments/run.py --list --status ready`
- `python numerical_experiments/run.py --all-ready --dry-run --mp_num_processors 2`
- `python -m pytest tests/test_cpu_limits.py -q` (`5 passed`)
- lost-sales binding smoke:
  - `invman_rust.lost_sales_reference_instance_names()`
  - `invman_rust.lost_sales_reference_costs("vanilla_l4_p4_poisson5")`
- help/import checks for:
  - `scripts/lost_sales/benchmark_canonical_suite.py`
  - `scripts/lost_sales/benchmark_full_suite.py`
  - `scripts/lost_sales/autoresearch_lost_sales.py`
  - `scripts/lost_sales/autoresearch_tree_structures.py`
  - `scripts/lost_sales/l4_demand_policy_suite.py`
  - `scripts/lost_sales/validate_reference_instance.py`
  - `scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py`
  - `scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`
  - `scripts/lost_sales_fixed_order_cost/autoresearch_fixed_order_cost.py`
  - `scripts/lost_sales_fixed_order_cost/autoresearch_fixed_order_tree_structures.py`
  - `scripts/lost_sales_fixed_order_cost/l4_demand_policy_suite.py`
  - `scripts/lost_sales_fixed_order_cost/validate_known_optimum.py`
  - `scripts/lost_sales_fixed_order_cost/diagnostics/analyze_policy.py`
  - `scripts/lost_sales_fixed_order_cost/diagnostics/benchmark_heuristics_grid.py`
  - `scripts/lost_sales_fixed_order_cost/diagnostics/compare_search_backends.py`
  - `scripts/dual_sourcing/benchmark_full_suite.py`
  - `scripts/one_warehouse_multi_retailer/run_paper_benchmark.py`
  - `scripts/multi_echelon/autoresearch_multi_echelon.py`
  - `scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py`
  - `scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py`
- `python -m py_compile scripts/lost_sales/compute_missing_heuristics_via_rust.py scripts/lost_sales/l4_demand_policy_suite.py scripts/lost_sales/validate_reference_instance.py`
- `python -m py_compile` for the migrated fixed-cost lost-sales utility and diagnostic scripts.
- migrated-test checks:
  - `python -m pytest tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py tests/test_numerical_experiments_catalog.py tests/test_cpu_limits.py -q` (`19 passed`; fixed-cost reference grid later expanded with baseline availability coverage)
  - `python -m pytest tests/test_soft_tree_policy.py tests/test_multi_echelon_problem.py -q` (`14 passed`)
  - `python -m pytest tests/test_reference_instance.py tests/test_policy_parameter_counts.py -q` (`2 passed`)
  - `python -m pytest tests/test_lost_sales_demand.py tests/test_fixed_order_cost_heuristics.py tests/test_fixed_order_cost_search_backends.py -q` (`8 passed`)
  - `python -m pytest tests/test_lost_sales_env.py -q` (`9 passed`)
  - `python -m pytest tests/test_invman_rust_bridge.py -q` (`17 passed`)
  - `python -m pytest tests/test_policy_factory.py -q` (`30 passed`)
  - `python -m pytest tests/test_dual_sourcing_problem.py -q` (`16 passed`)
  - combined migrated set: `python -m pytest tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py tests/test_numerical_experiments_catalog.py tests/test_cpu_limits.py tests/test_soft_tree_policy.py tests/test_multi_echelon_problem.py tests/test_policy_parameter_counts.py tests/test_reference_instance.py tests/test_lost_sales_demand.py tests/test_fixed_order_cost_heuristics.py tests/test_fixed_order_cost_search_backends.py tests/test_lost_sales_env.py tests/test_invman_rust_bridge.py tests/test_policy_factory.py tests/test_dual_sourcing_problem.py -q` (`115 passed`)
  - fixed-cost baseline/detail/trace regression: `python -m pytest tests/test_fixed_order_cost_reference_grid.py tests/test_fixed_order_cost_heuristics.py tests/test_fixed_order_cost_search_backends.py -q` (`14 passed`; latest run 2026-06-04, 0.67s)
  - catalog/launcher/CPU regression: `python -m pytest tests/test_numerical_experiments_catalog.py tests/test_cpu_limits.py -q` (`14 passed`; latest run 2026-06-04, 1.10s)
  - CPU-limit regression: `python -m pytest tests/test_cpu_limits.py -q` (`5 passed`; latest run 2026-06-04, 0.04s)
  - descriptor action-binding regression: `python -m pytest tests/test_soft_tree_policy.py tests/test_policy_factory.py -q` (`41 passed`; latest run 2026-06-04, 0.13s)
  - learned-policy trace regression: `python -m pytest tests/test_lost_sales_env.py tests/test_fixed_order_cost_heuristics.py -q` (`16 passed`; latest run 2026-06-04, 1.75s)
  - fixed-cost diagnostic archive regression: `python -m pytest tests/test_fixed_order_cost_heuristics.py -q` (`5 passed`; latest run 2026-06-04, 2.12s)
  - Rust-first migration-layout regression: `python -m pytest tests/test_rust_first_migration.py -q` (`5 passed`; latest run 2026-06-04, 0.31s)
  - migration-layout + ready-catalog regression: `python -m pytest tests/test_rust_first_migration.py tests/test_numerical_experiments_catalog.py -q` (`14 passed`; latest run 2026-06-04, 1.15s)
  - migration-layout + practical-report + ready-catalog regression: `python -m pytest tests/test_practical_benchmark_reports.py tests/test_rust_first_migration.py tests/test_numerical_experiments_catalog.py -q` (`15 passed`; latest run 2026-06-04, 1.22s)
  - full Python test suite: `python -m pytest tests -q` (`147 passed`; latest run 2026-06-04, 49.70s)
  - `python -m py_compile tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py tests/test_soft_tree_policy.py tests/test_multi_echelon_problem.py tests/test_reference_instance.py tests/test_policy_parameter_counts.py tests/test_lost_sales_demand.py tests/test_fixed_order_cost_heuristics.py tests/test_fixed_order_cost_search_backends.py tests/test_lost_sales_env.py tests/test_invman_rust_bridge.py tests/test_policy_factory.py tests/test_dual_sourcing_problem.py`
- migrated helper-script checks:
  - `python -m py_compile autoresearch/fixed_cost_ordinal_stability/replay_exact_ordinal.py autoresearch/fixed_cost_ordinal_stability/ablate_state_drift.py scripts/raw_state_single_policy_probe.py scripts/evaluate_saved_policy.py scripts/perishable_inventory/common.py scripts/joint_pricing_inventory/common.py`
  - `python -m py_compile scripts/random_yield_inventory/common.py scripts/procurement_removal_inventory/common.py scripts/random_yield_inventory/train_soft_tree_reference.py scripts/procurement_removal_inventory/train_soft_tree_reference.py`
  - `python autoresearch/fixed_cost_ordinal_stability/replay_exact_ordinal.py --help`
  - `python scripts/raw_state_single_policy_probe.py --help`
  - `python scripts/evaluate_saved_policy.py --help`
  - `python scripts/random_yield_inventory/train_soft_tree_reference.py --help`
  - `python scripts/procurement_removal_inventory/train_soft_tree_reference.py --help`
  - perishable/joint-pricing smoke snippets instantiate current `Policy` soft trees and produce Rust rollout kwargs
  - random-yield/procurement-removal smoke snippets instantiate current `Policy` soft trees and run one-seed Rust rollouts
- stale import scan:
  - `rg "invman\\.policies|invman\\.problems\\.lost_sales|invman\\.problems\\.lost_sales_fixed_order_cost" scripts/lost_sales scripts/lost_sales_fixed_order_cost -g '*.py'`
  - returns no matches
  - `rg "invman\\.problems|invman\\.policies" tests -g '*.py'`
  - returns no matches
  - `rg "invman\\.problems|invman\\.policies" tests scripts/lost_sales scripts/lost_sales_fixed_order_cost scripts/dual_sourcing -g '*.py'`
  - only finds explanatory docstring text in `scripts/dual_sourcing/dual_sourcing_benchmark_lib.py`, not live imports
- tiny Rust-routed validator smoke:
  - `python scripts/lost_sales/validate_reference_instance.py --horizon 100 --num_seeds 1 --tolerance 999`
- exact fixed-cost known-optimum validator:
  - `python scripts/lost_sales_fixed_order_cost/validate_known_optimum.py`
  - reproduces Bijvank 2015 Table 1 tightly: optimal 11.46305 vs published 11.46; modified (s,S,q) 11.49740 vs published 11.50
- fixed-cost diagnostic availability checks:
  - `python scripts/lost_sales_fixed_order_cost/diagnostics/benchmark_heuristics_grid.py --limit 1 --search_horizon 100`
  - `python scripts/lost_sales_fixed_order_cost/diagnostics/compare_search_backends.py --search_horizon 100 --eval_horizon 100 --eval_seeds 1`
- fixed-cost detailed binding smoke:
  - `invman_rust.lost_sales_fixed_heuristics_all_detailed(...)`
  - returns per-policy `params`, `mean_cost`, `top`, and `evaluated_candidates`
- fixed-cost trace binding smoke:
  - `invman_rust.lost_sales_fixed_policy_trace_from_demands(...)`
  - returns policy metadata, `mean_cost`, `warm_up_periods`, and per-period trace rows
- learned-policy supplied-demand trace binding smoke:
  - `invman_rust.lost_sales_soft_tree_trace_from_demands(...)`
  - `invman_rust.lost_sales_linear_trace_from_demands(...)`
  - `invman_rust.lost_sales_nn_trace_from_demands(...)`
  - return policy metadata, `mean_cost`, `warm_up_periods`, and per-period state/action/cost trace rows
- policy descriptor action binding smoke:
  - `invman_rust.soft_tree_action_vector_from_flat_params(...)`
  - `invman_rust.linear_policy_action_from_flat_params(...)`
  - `invman_rust.nn_policy_action_from_flat_params(...)`
- fixed-cost diagnostic checks after trace/action wiring:
  - `python scripts/lost_sales_fixed_order_cost/diagnostics/analyze_policy.py --help`
  - in-memory zeroed linear `Policy` smoke for `coarse_grid_action_histogram(...)` returns a non-empty Rust-routed histogram
  - end-to-end temporary saved-policy CLI smoke: `python scripts/lost_sales_fixed_order_cost/diagnostics/analyze_policy.py --model_dir <tmp_policy> --horizon 5 --trace_horizon 4 --trace_rows 2`
  - the CLI smoke emits learned-policy trace rows, searched modified `(s,S,q)` params/trace rows, and a non-empty coarse state-grid action histogram
  - the CLI smoke with `--output_json <tmp_json>` writes the same diagnostic trace payload that it prints to stdout
- dual-sourcing autoresearch factor-screen import/help check:
  - `python -m py_compile autoresearch/dual_sourcing_policy_search/run_factor_screen.py`
  - `python autoresearch/dual_sourcing_policy_search/run_factor_screen.py --help`
- broad live-import scan:
  - `rg '^(from|import) invman\\.(problems|policies)' -g '*.py'`
  - returns no matches after the helper migration/deprecation slice
  - stricter embedded-import scan `rg '^\\s*(from|import) invman\\.(problems|policies)' -g '*.py'`
  - returns no matches after the random-yield/procurement-removal common-helper migration
  - active nested-Rust path scan `rg 'rust/src|rust/Cargo|--manifest-path rust|PACKAGE_ROOT\\s*/\\s*["\\x27]rust["\\x27]|Path\\([^\\n]*["\\x27]rust["\\x27]' README.md AGENTS.md docs autoresearch scripts numerical_experiments src paper -g '*.md' -g '*.py' -g '*.rs' -g '*.tex' -g '*.toml'`
  - returns no matches after the ready-surface/root-crate sweep
  - live deleted-package import scan `rg '^\\s*(from|import)\\s+invman\\.(problems|policies)' README.md AGENTS.md docs autoresearch scripts numerical_experiments tests src paper -g '*.md' -g '*.py' -g '*.rs' -g '*.tex'`
  - returns no matches; remaining `invman.problems.*` / `invman.policies.*` text is historical warning text, not live imports
  - `rg 'pending_rust_binding|pending the Rust grid|fixed-cost heuristic baseline binding is pending|fixed-cost \\(s,S,q\\) baselines pending' scripts tests` returns no matches
- ready-suite runnable-surface checks:
  - automated catalog tests confirm every ready suite command is repo-local, avoids deleted nested-Rust paths, and every ready suite script accepts `--help` under a one-worker CPU cap
  - `python numerical_experiments/run.py --all-ready --dry-run --mp_num_processors 2`
  - launcher unit test confirms selected suite subprocesses receive a CPU-limited environment
  - `python scripts/one_warehouse_multi_retailer/run_paper_benchmark.py --instance_names kaynov2024_instance_1 --training_episodes_small 1 --training_episodes_large 1 --es_population 2 --train_seed_batch 1 --heuristic_search_replications 1 --benchmark_replications 1 --eval_seeds 1 --mp_num_processors 1 --artifact_dir /tmp/invman_owmr_smoke_artifacts --output_json /tmp/invman_owmr_smoke.json --output_markdown /tmp/invman_owmr_smoke.md`
  - tiny OWMR smoke completes and records `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` as `1`
- exploratory CPU-audit checks:
  - `python -m py_compile scripts/run_lost_sales_paper_benchmarks.py scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py scripts/perishable_inventory/autoresearch_perishable_inventory.py scripts/joint_replenishment/benchmark_learned_vs_heuristics.py scripts/joint_replenishment/autoresearch_joint_replenishment.py scripts/joint_replenishment/evaluate_setting5_vs_vi_optimum.py scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py scripts/one_warehouse_multi_retailer/benchmark_learned_vs_heuristic.py`
  - `python -m py_compile scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py scripts/joint_replenishment/benchmark_learned_vs_heuristics.py`
  - `python scripts/run_lost_sales_paper_benchmarks.py --dry_run --limit 1 --mp_num_processors 2`
  - help checks pass for the patched ameliorating, general-backorder, serial, perishable, joint-replenishment, OWMR, vendor-managed, and production-assembly runners
  - `rg 'set RAYON_NUM_THREADS|RAYON_NUM_THREADS already exported|os\\.environ\\.setdefault\\(\"(RAYON_NUM_THREADS|OMP_NUM_THREADS)|for _var in \\(\"RAYON_NUM_THREADS' scripts autoresearch numerical_experiments invman -g '*.py'` returns no active matches
- practical helper migration checks:
  - `python -m py_compile invman/benchmarks/__init__.py invman/benchmarks/practical.py scripts/perishable_inventory/run_paper_benchmark.py scripts/perishable_inventory/run_practical_benchmark.py scripts/nonstationary_lot_sizing/run_practical_benchmark.py tests/test_rust_first_migration.py`
  - `python -m py_compile tests/test_practical_benchmark_reports.py tests/test_rust_first_migration.py invman/benchmarks/practical.py`
  - `python scripts/perishable_inventory/run_paper_benchmark.py --help`
  - `python scripts/perishable_inventory/run_practical_benchmark.py --help`
  - `python scripts/nonstationary_lot_sizing/run_practical_benchmark.py --help`
  - `python scripts/perishable_inventory/run_practical_benchmark.py --output_json /tmp/invman_perishable_practical_smoke.json --output_markdown /tmp/invman_perishable_practical_smoke.md`
  - `python scripts/nonstationary_lot_sizing/run_practical_benchmark.py --output_json /tmp/invman_nonstationary_practical_smoke.json --output_markdown /tmp/invman_nonstationary_practical_smoke.md`

Resolved items and remaining caveats:

- the broader Python test migration is now clear for the current `tests/` tree: the full Python suite passes with the new migration/practical-report regressions included, and no test imports deleted `invman.problems.*` or `invman.policies.*` paths.
- fixed-cost lost-sales grid heuristic baseline costs, winning params, top candidates, supplied-demand fixed-policy traces, learned-policy supplied-demand traces, and learned-policy coarse-grid action summaries are now wired through Rust bindings. The remaining optional paper-output detail is whether to include trace artifacts in tables/appendices, not whether the binding exists.
- remaining deleted-package references are explicit explanatory "removed path" notes, not live Python imports or active usage instructions; active nested-Rust path references now scan clean and have regression coverage.
- the canonical Rust test command is now `cargo test --manifest-path Cargo.toml -q`; Python extension builds remain `python scripts/build_rust_extension.py`, which explicitly enables the PyO3 extension feature.

## Latest Limiting Factors

1. Post-migration import fallout is cleared for the current Python test suite and the stable lost-sales/fixed-cost/dual-sourcing lanes. The full `tests/` suite passes under the flattened API. The previously stale fixed-cost/lost-sales helper scripts, saved-policy evaluator, perishable/joint-pricing helpers, random-yield/procurement-removal helpers, and practical benchmark report helpers have been migrated or deliberately archived. Remaining deleted-package references are explicit removed-path warnings, not live Python imports or active usage instructions.

2. The compatibility direction is now effectively migration/deprecation, not broad shims: stable lanes use `invman.policy`, `invman.policy_registry`, `invman.rollout_fitness`, benchmark helper builders, and `invman_rust`. Add old-path shims only if a specific preserved artifact requires them.

3. Rust verification is now split cleanly from Python extension builds. Default Cargo features are Rust-native and `cargo test --manifest-path Cargo.toml -q` passes; extension builds must go through `python scripts/build_rust_extension.py`, which enables the `python-extension` feature for maturin/PyO3.

4. Installed-extension/source drift is possible. The active environment exposes the new bindings, but the Rust source is dirty. After Rust edits, rebuild with `python scripts/build_rust_extension.py` and record which source commit/artifact produced benchmark numbers.

5. Benchmark readiness is improved but still needs discipline. The catalog lists ready suites, lost-sales/fixed-cost/dual-sourcing scripts import, the full Python suite passes, active source surfaces now have a root-Rust path regression, and fixed-cost grid-wide heuristic costs, winning params, fixed-policy traces, learned-policy supplied-demand traces, and learned-policy coarse-grid action summaries now come from Rust. Trace diagnostics can be archived through `--output_json`; remaining publication-grade work is deciding whether those trace artifacts belong in final paper tables/appendices.

6. CPU oversubscription is patched for the stable lost-sales/fixed-cost full-suite subprocess paths, the top-level numerical-experiment launcher, the ready OWMR paper benchmark, and the audited exploratory runners that previously carried manual Rayon/OpenMP cap code. Remaining CPU work is evidence-gathering on full-budget launcher behavior rather than obvious ad hoc env setup.

7. Reporting discipline is now a hard constraint. `literature_verified` should only be set when repo exact/heuristic code reproduces a published number. Published DRL/A3C/PPO rows are comparison rows, not repo-verified algorithms.

8. Dual sourcing is still mainly policy-geometry limited. Current evidence favors factorized capped-delta / capped-dual-index coordinates, with row-conditioned geometry: axis-linear for `l_r = 2`, tighter axis-constant small-cap trees for `l_r in {3,4}`. `autoresearch/dual_sourcing_policy_search/run_factor_screen.py` has been migrated to current policy-registry and Rust-backed reference helpers; remaining work is final policy-screen evidence, not old-import repair.

9. Serial Clark-Scarf is match-only. The comparator is a true optimum, so the learned-policy result can tie the optimum within simulation error but should never be framed as beating it.

10. Faithful ameliorating inventory is bound-limited. The current single-purchase learned policy can beat the simple order-up-to heuristic, but it remains far below the perfect-information LP upper bound and is not comparable to the paper's full three-part-action DRL gap.

11. Several exploratory families still carry honest blockers: missing published anchors, self-consistency-only exact checks, documentation drift, or learned policies that still lose to tuned heuristics. Do not promote them into headline claims until their runners and verification status are clean.

12. The paper workspace is in heavy churn. There are deleted docs, a deleted old paper file, a new untracked paper file, regenerated figures/PDFs, and updated manuscript claims. Freeze the source of truth before doing final benchmark or Overleaf work.

## Rust-First Migration

Target direction: make the Rust crate the main project surface instead of a nested `rust/` folder. The repo root now carries the Rust crate entrypoints (`Cargo.toml`, `Cargo.lock`, `src/`), with Python kept as a tier-two support layer for bindings, experiments, papers, and orchestration.

Proposed target layout:

- `Cargo.toml`, `Cargo.lock`, `src/`: primary Rust crate at repo root
- `src/problems/`, `src/core/`, `src/case_studies/`: tier-one Rust source domains
- `python/invman/` or `bindings/python/invman/`: tier-two Python API, CMA-ES runners, and compatibility shims
- `scripts/`, `numerical_experiments/`, `paper/`, `docs/`, `autoresearch/`: tier-two project support surfaces
- old `rust/`: removed; only root `Cargo.toml`, `Cargo.lock`, `src/`, and tier-two support folders remain

Migration steps:

1. Done: move `Cargo.toml`, `Cargo.lock`, and `src/` to the repo root equivalents.
2. Done: update `scripts/build_rust_extension.py`, setup/build docs, and any `--manifest-path Cargo.toml` references to use root `Cargo.toml`.
3. Done for the broad text/code path sweep: update references from the old nested crate path to `src/...`.
4. Done for this migration slice: keep `invman/` at the root as the tier-two support package, and move the old maturin `pyproject.toml` to `bindings/python/invman_rust/pyproject.toml` so root `setup.py` still owns `pip install -e .` for `invman`.
5. Done: rebuild `invman_rust` through the canonical maturin path and confirm the installed extension exposes the lost-sales binding surface.
6. Done for the first canary: catalog listing, CPU tests, `cargo check`, focused lost-sales Rust-binding smoke, and import checks for ready lost-sales benchmark scripts.
7. Done: remove the empty/obsolete `rust/` folder after docs and scripts stopped depending on it.

Migration risks:

- Path churn can invalidate benchmark scripts, docs, paper references, and Overleaf push assumptions.
- Python editable-install metadata may need a new package layout if `invman/` moves under `python/`.
- Cargo/PyO3 verification depends on the feature split: Rust tests use default features, while Python extension builds use the explicit `python-extension` feature through `scripts/build_rust_extension.py`.
- Existing untracked/generated artifacts should not be moved blindly; separate source moves from output cleanup.

## Proposed Active Migration Goal

Goal: migrate `invman` to a Rust-first repository layout with the Rust crate at the repo root, while preserving benchmark reproducibility and using stable lost-sales checks as the first canary.

Important sequencing decision: do not physically move only `lost_sales` first. Rust module resolution is anchored at the crate root (`src/lib.rs` and `src/problems/mod.rs`), so moving one problem family while leaving `Cargo.toml` under `rust/` would create an awkward split crate. The safer first implementation slice is:

- moved the crate root once (`Cargo.toml` + `src/` now live at the repo root)
- update only the minimum build and path references required to compile/import
- validate with the stable lost-sales surface before touching newer/dirty problem families semantically

Stage 0: preflight and freeze

- Record current dirty state and avoid mixing source relocation with generated-output cleanup.
- Confirm `invman_rust` build route through `python scripts/build_rust_extension.py`.
- List stale references with:
  - `rg "rust/src|rust/Cargo|--manifest-path rust" ...`
  - `rg "invman\\.problems|invman\\.policies" ...`

Stage 1: root the Rust crate

- Done:
  - `Cargo.toml` is at the repo root
  - `Cargo.lock` is at the repo root
  - `src/` is at the repo root
  - the old maturin `pyproject.toml` is under `bindings/python/invman_rust/`
- Python, scripts, docs, paper, and benchmark outputs stayed in place for this stage.
- `scripts/build_rust_extension.py` now uses root `Cargo.toml`.

Stage 2: lost-sales canary

- Update references needed by the stable lost-sales path:
  - docs and scripts that mention `src/problems/lost_sales/...`
  - any build/test command using `--manifest-path Cargo.toml`
  - README/AGENTS examples that still point to deleted Python problem-package files
- Run the minimum canary:
  - done: `python scripts/build_rust_extension.py`
  - done: `cargo check --manifest-path Cargo.toml -q`
  - done: `cargo test --manifest-path Cargo.toml -q`
  - done: `python -m pytest tests/test_cpu_limits.py -q`
  - done: `python numerical_experiments/run.py --list`
  - done: import/help checks for `scripts/lost_sales/benchmark_full_suite.py` and `scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`
  - done: `invman_rust` lost-sales reference binding smoke
  - done: migrate and import-check vanilla lost-sales autoresearch/utility scripts away from deleted `invman.problems.*` and `invman.policies.*`
  - done: migrate and import-check fixed-cost lost-sales autoresearch/utility/diagnostic scripts away from deleted `invman.problems.*` and `invman.policies.*`
  - done: migrate focused vanilla lost-sales env/demand tests to public Rust bindings and reference tables
  - done: migrate fixed-cost lost-sales heuristic/search-backend tests to public Rust bindings
  - done: migrate the lost-sales Rust bridge tests to the current `Policy` descriptor plus `invman.rollout_fitness`
  - done: migrate the policy-factory tests to current `Policy` descriptor assertions and explicit unsupported dense dual-sourcing behavior
  - done: migrate dual-sourcing reference/grid/search/rollout tests to current Rust bindings and `scripts.dual_sourcing.dual_sourcing_benchmark_lib`
  - done: migrate `autoresearch/dual_sourcing_policy_search/run_factor_screen.py` to the current policy registry and Rust-backed dual-sourcing reference/heuristic helpers
  - done: migrate fixed-cost/lost-sales helper probes and saved-policy evaluation to Rust-routed flattened APIs
  - done: migrate perishable and joint-pricing common soft-tree helpers from removed `SoftTreePolicy` imports to current `Policy`
  - done: migrate random-yield and procurement-removal common soft-tree helpers from removed `SoftTreePolicy` imports to current `Policy`
  - done: archive the fixed-cost state-drift ablation script that depends on deleted env/model APIs
  - done: expose fixed-cost heuristic winning params/top candidates through `lost_sales_fixed_heuristics_all_detailed` and route benchmark summaries to it
  - done: expose fixed-cost fixed-policy per-period traces through `lost_sales_fixed_policy_trace_from_demands`, add Python trace regression coverage, and route diagnostic heuristic action summaries through a bounded trace sample
  - done: expose Rust action-evaluation bindings for current `Policy` descriptors and route the fixed-cost diagnostic's learned-policy coarse-grid action histogram through them
  - done: expose Rust supplied-demand per-period traces for learned lost-sales soft-tree, linear, and NN policies, add regression coverage, and route fixed-cost diagnostic learned-policy trace samples through them

Stage 3: Python tier-two cleanup

- Current decision for this slice: keep `invman/` at repo root as the tier-two Python support package.
- Current old-import policy: migrate runnable scripts to `invman.policy`, `invman.policy_registry`, `invman.rollout_fitness`, benchmark helper builders, and `invman_rust`; deliberately archive scripts that require deleted Python env/model internals.
- Active-surface README/docs references have been swept for the stable lanes. Remaining old-path text is intentionally explanatory/historical.
- Current ready catalog scripts expose `--help` under CPU caps, ready commands are repo-local and avoid deleted nested-Rust paths, and the top-level ready dry-run works. Do not claim full benchmark completion from that alone because most ready suites are intentionally long-running and still need full-budget execution for final numbers.

Stage 4: sweep non-lost-sales references

- Done for active nested-Rust path references: README/agent/docs/autoresearch/scripts/numerical-experiment/source/paper scans return no live `rust/src`, `rust/Cargo`, `--manifest-path rust`, or root `rust` path construction references. Remaining old-path text is intentionally explanatory/historical.
- Use problem-family order based on churn:
  - stable first: `lost_sales`, `lost_sales/fixed_order_cost`
  - then already-routed families: `dual_sourcing`, `multi_echelon`
  - then newer dirty families: `ameliorating_inventory`, `multi_echelon/serial`, exploratory families

Stage 5: verification and cleanup

- Done: define the canonical Rust verification command after the PyO3/root move. Cargo default features are Rust-native, `cargo test --manifest-path Cargo.toml -q` passes, and the Python extension build route explicitly enables `python-extension`.
- Done: remove the obsolete `rust/` folder after no scripts/docs/tests referenced it.
- Done for the ready launcher surface: `numerical_experiments/run.py` propagates CPU limits, all ready suite commands dry-run, ready scripts accept `--help` under CPU caps, ready commands are checked for deleted nested-Rust paths, and the OWMR ready script no longer writes into deleted `rust/src/...` defaults.
- Continue updating `plan.md`, README files, and `numerical_experiments/catalog.py` as runnable state and verification status change.

## Next Work

1. Decide whether final paper tables/appendices should include fixed-cost heuristic trace artifacts or learned-policy trace artifacts; params, top candidates, supplied-demand fixed-policy traces, supplied-demand learned-policy traces, learned-policy coarse-grid action summaries, and archiveable diagnostic JSON are now available.
2. Use the catalog help/dry-run/CPU-env checks as the gate before promoting more suites to "ready"; the obvious manual Rayon/OpenMP setup has been cleared.
3. Run full-budget ready suites only when final benchmark numbers are needed; current evidence proves import/help/dry-run plus targeted smoke, not full-paper numerical completion.
