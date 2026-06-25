# Verification Ledger — invman benchmark repo

**Honest status, audited 2026-06-06.** This ledger records, per problem (or sub-family), the published number, what was reproduced by **actually re-running** the env/solver this audit, the verdict, and the re-run method. It separates three distinct things that are easy to conflate:

- **A published *peer-reviewed* number reproduced by re-run** (the gold bar).
- **A reference-implementation / companion-code number reproduced by re-run** (e.g. stockpyl, author CSVs) — honest, but not a paper table.
- **A repo-native self-consistency anchor reproduced by re-run** (an exact DP the repo itself defines) — a correctness check, *not* a literature claim.

## One-paragraph honest summary

Of the 14 systems, **5 are genuinely literature-verified-by-rerun against a peer-reviewed published number** (lost_sales, dual_sourcing, perishable_inventory, plus the serial + general_backorder_fixed_cost sub-families of multi_echelon, plus spare_parts_inventory's *adjacent* Kranenburg module). **2 more reproduce a reference-implementation / companion-code or closed-form number by re-run but not a peer-reviewed article table** (nonstationary_lot_sizing against the author's testbed CSVs; decentralized_inventory_control's closed-form board-game 204; joint_replenishment reproduces a published *action* q=(0,6) but no published cost exists). **4 are faithful_unverified** — the env is structurally faithful and only repo-native self-consistency anchors re-ran (ameliorating_inventory, joint_pricing_inventory, procurement_removal_inventory, random_yield_inventory). **The trainable env of spare_parts_inventory and vendor_managed_inventory, and the env.rs MDP of decentralized_inventory_control and the general-network slice of padn, are faithful-but-unverified** (no published cost reproduced by the *trainable* env, even though an adjacent module is verified). There are **no heuristic_only systems** (all carry at least one anchor). There are **4 standing snapshot_only_not_rerun verification debts** that assert carried==published literals without executing the env, plus several minutes-scale rows validated on a prior date but not re-run this audit; each is listed below and must be converted into an executing re-run assertion.

## Counts by status

| Status | Count | Systems / sub-families |
|---|---|---|
| verified_rerun (peer-reviewed published number) | 5 | lost_sales; dual_sourcing; perishable_inventory; multi_echelon/serial; multi_echelon/general_backorder_fixed_cost(set1+KT); spare_parts_inventory/Kranenburg-module |
| verified_rerun (reference-impl / companion / closed-form, not a paper table) | 3 | nonstationary_lot_sizing (author CSVs); joint_replenishment (published *action*); decentralized_inventory_control (closed-form 204) |
| faithful_unverified (only repo-native self-consistency re-ran; or env faithful but no published cost) | 6 | ameliorating_inventory (trainable env; **LP bound now verified_rerun vs companion, see below**); joint_pricing_inventory; procurement_removal_inventory; random_yield_inventory; spare_parts_inventory/trainable-env; vendor_managed_inventory |
| snapshot_only_not_rerun (asserted literals, NOT executed — DEBT) | **3 items** | divergent A3C relative rows; gbk set2/set3 (+223%, **now honestly flagged `literature_verified=false`+comment, 2026-06-06**); van_oers 2024 Table 1 (padn/spare-parts adjacent) |
| verified-but-not-re-run-this-audit (slow real reproduction, NOT a snapshot literal) | 1 | dual_sourcing l_r=3,4 (real 37^L value-iteration reproduction, ~30–40 min total; validated 2026-06-04 within 0.01pp of Fig 9; left as on-demand, not snapshot) |
| **debt CLOSED 2026-06-06** | 1 | ameliorating_inventory perfect-info LP bound — new binding `ameliorating_inventory_perfect_info_lp_bound_summary` re-runs the bound to ~1e-8 vs companion (was Rust-only) |
| no_published_number | 4 | (subset of faithful_unverified) joint_pricing_inventory, procurement_removal_inventory, random_yield_inventory, vendor_managed_inventory all lack a public per-instance number to target |

Note: counts span sub-families, so the column sums exceed 14. multi_echelon and spare_parts_inventory are intentionally split because their sub-families have *different* verdicts.

---

## Group 1 — verified_rerun against a PEER-REVIEWED published number

| Problem / sub-family | Published number | Re-run reproduced value | Status | How re-run |
|---|---|---|---|---|
| **lost_sales / fixed_order_cost** | Bijvank 2015 Table 1: optimal 11.46, (s,S) 11.62, (s,nQ) 11.56, mod (s,S,q) 11.50 | Exact DP: optimal 11.4631 (gap +0.0031), (s,S) 11.6181, (s,nQ) 11.5552, mod 11.4974 (all <0.005) | verified_rerun (genuine EXACT solver) | `ir.lost_sales_fixed_order_cost_exact_literature_summary('bijvank2015_table1_l2_p14_k5',24)` |
| **lost_sales / vanilla** | Zipkin 2008 Table 3a: myopic 5.06, myopic2 4.82, SVBS 5.83 (optimal 4.73 carried, not recomputed) | myopic1 5.0569, myopic2 4.8208, svbs 5.8153 (within ~0.015) | verified_rerun (3 heuristic rows; optimum is a carried Zipkin value) | `ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995)` |
| **dual_sourcing (l_r=2 rows)** | Gijs Fig 9 gaps dual_l2_ce105: CDI 0.00 / DI 0.11 / SI 0.56 / TBS 0.06 | optimal 216.770; CDI 0.0058 / DI 0.1164 / SI 0.5675 / TBS 0.0615 (all <=0.0075pp) | verified_rerun (bounded-DP gaps; no published *absolute* cost exists) | `invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce105',...)` |
| **dual_sourcing (CDI-optimality 3-tier taxonomy, 2026-06-07)** | (repo-native, no published number) | Validated-box bounded-DP sweep over premium/penalty/CV at l_r=2 → CDI is heuristics-excellent across the whole reachable regime: max gap-to-optimum **+0.305% single-path / +0.160% OOS** at Tier-C `dual_l2_ce110_b50_u08_catC` (U[0,8]); **no >=5% hard regime exists (honest negative)**. Tiers: **A** CDI-optimal (≤0.12%, the 6 Gijs rows) / **B** moderate (~0.12–0.20%, `dual_l2_ce110_b50_u04_catB`) / **C** hardest (`..._u08_catC`). Seed-robust learned-vs-CDI on Tier-C: **+0.615% ± 0.346%, 0/5 beat → robust-LOSS** (CDI wins). | repo-native taxonomy probe (bounded-DP optimum denominator; NOT a published row) | `dual_sourcing_bounded_average_cost_optimal_summary(...)`; `scripts/dual_sourcing/cdi_gap_to_optimum_regime_sweep.py`, `cdi_out_of_sample_gap_to_optimum.py`, `seed_robust_learned_vs_cdi_tier_c.py`; doc `docs/benchmarks/DUAL_SOURCING_INSTANCE_TAXONOMY_2026_06_07/README.md` |
| **perishable_inventory (m=2/L=1)** | Farrington 2025 VI FIFO -1457, LIFO -1553; De Moor S=7/S=5 + 9x9 policy tables; base-stock FIFO -1474 | VI -1457.281 / -1552.991; S=7 / S=5; policy tables match=True; FlowNet base-stock -1475 (tol 1.0) | verified_rerun (genuine VI re-derivation of 3 independent quantities) | `ir.perishable_inventory_exact_mdp_summary(...)`; `ir.perishable_inventory_flownet_policy_verification_summary()` |
| **multi_echelon / serial** | Snyder & Shen Ex6.1 optimal 47.65 | 47.6654 (Ex6.1); stockpyl Poisson N1/N2/N3 4.2211/16.7983/72.0467; 5-stage 225.867/226.846 | verified_rerun (TRUE Clark-Scarf optimum) | `ir.multi_echelon_serial_exact_normal_solution([3,2,2],[1,1,2],37.12,5,1)` |
| **multi_echelon / general_backorder_fixed_cost (set1, KT)** | Geevers set1 10467; Kunnumkal-Topaloglu 4059 | set1 10384.9 (-0.78%); KT 3933.3 (-3.1%) | verified_rerun (within published band) | `ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('geevers2023_general_set1',replications=200,seed=1234)` |
| **spare_parts_inventory / Kranenburg module** (⚠ structurally DIFFERENT from the trainable env) | Kranenburg 2006 Table 5.2: R1*=9.09 C1=91.90, R3*=6.10 C3=63.00 (35 rows) | 35/35 within 0.02; base case R1*=9.0900/91.9000, R3*=6.1000/63.0000 | verified_rerun (analytical exact, adjacent sub-family only) | loop `spare_parts_inventory_kranenburg_exact_summary(name)` over `spare_parts_inventory_kranenburg_reference_instances()` |

---

## Group 2 — verified_rerun against a reference-implementation / companion / closed-form number (NOT a peer-reviewed article table)

| Problem | "Published" source | Re-run reproduced value | Status | How re-run |
|---|---|---|---|---|
| **nonstationary_lot_sizing** | Author **companion-code testbed CSVs** (HenriDeh/DRL_MMULS), NOT the EJOR article. constant_10: simple (s,S) 1832.914, rolling-DP 1711.741 | simple (s,S) 1834.918 (+0.109%), rolling-DP 1714.148 (+0.141%); (s,S) levels EXACT; rolling-DP first-period (28,42) EXACT | verified_rerun vs companion CSV (repo flag literature_verified=false is honest) | `nonstationary_lot_sizing_simple_s_s_levels(...)`, `..._simulate_policy(...)`, `..._simulate_rolling_dp_policy(...)` |
| **joint_replenishment** | Vanvuchelen 2020 Fig-3 optimal **ACTION** q=(0,6) at state (5,0); NO published cost table exists | VI greedy action (5,0) = (0,6), converged iter 2260; finite-horizon DP comparator matches README literals (266.386 etc.) | verified_rerun of a published *action*; cost numbers are repo-native | `ir.joint_replenishment_exact_dp_summary()`; `benchmark_vanvuchelen_settings.value_iteration_setting5(...)` |
| **decentralized_inventory_control** (closed-form port only) | Sterman/Edali-Yasarcan anchor-and-adjust [46,50,54,54] total 204 (the value the R-port emits, NOT a transcribed table) | closed-form board-game [46,50,54,54]/204 EXACT; Clark-Scarf constant-demand 0.0 | verified_rerun (closed-form only); **env.rs yields 378/278 — see Group 4** | `decentralized_inventory_control_classic_sterman_literature_summary()` |

---

## Group 3 — faithful_unverified (env faithful; only repo-native self-consistency re-ran, or no published number to target)

| Problem | Published number | Re-run reproduced value | Status | How re-run |
|---|---|---|---|---|
| **ameliorating_inventory** | Perfect-info LP bounds: spirits_0001 1991.934, port_wine 2444.801, spirits_1002 1663.889 | **LP bound NOW re-run (2026-06-06): 1991.9344293931 / 2444.801, reproduces companion to ~1e-8** via new binding. Learned env path re-ran (spirits_0001 smoke 77.96 vs gate 20.07). | LP bound = **verified_rerun (companion code)**; trainable env = faithful_unverified (no published achieved cost) | bound: `python -c "import invman_rust as ir; print(ir.ameliorating_inventory_perfect_info_lp_bound_summary('pahr_grunow2025_spirits_0001'))"`; learned: `autoresearch_ameliorating_inventory_average_profit.py --instance spirits_0001 --budget smoke` |
| **joint_pricing_inventory** | NONE (no public per-instance optimal-profit number) | exact DP optimal -33.178121049724 (2,1) reproduced incl. independent Python DP within 1e-9; critical-fractile y*=(3,2,2) matched | faithful_unverified / no_published_number | `ir.joint_pricing_inventory_exact_dp_summary()`; env brute force via `..._step` |
| **procurement_removal_inventory** | NONE (2017 numbers pricing-coupled NPV ~84000; 2025 qualitative only) | exact-DP optimal 31.78026111369698 to 1e-10; interval_stock (6,6) 358.107, (4,9) 244.117 reproduced exactly | faithful_unverified / no_published_number | `r.procurement_removal_inventory_exact_dp_summary()`; `..._simulate_policy(...)` |
| **random_yield_inventory** | NONE usable (Yan 2026 / Chen 2018 paywalled; Inderfurth 2015 different yield model) | exact-DP slice optimal 40.0599, LIR 47.7138, WNH 60.3936 reproduced to full precision; sim LIR 203.619, WNH 222.436 | faithful_unverified / no_published_number (self-consistent only) | `invman_rust.random_yield_inventory_exact_dp_summary()`; `..._policy_discounted_cost_summary(...)` |
| **spare_parts_inventory / trainable env** | NONE (no paper publishes a cost for this exact periodic-review repairable construction) | repo-native exact DP optimal 28.39366 weakly dominates base_stock 28.39366 and lead_time_mean_cover 28.91225 | faithful_unverified (the RL-relevant env; Kranenburg in Group 1 is a *different* model) | `m.spare_parts_inventory_exact_dp_summary()` |
| **vendor_managed_inventory** | NONE peer-reviewed (paper table paywalled). Only OPEN numbers are an instructor *handout* (S=15/31.53/26.96) | handout newsvendor MDH 15.0 / six-sigma 31.53122 / newsvendor 26.99054 reproduced; env step period_cost=6.0 matches; low_penalty heuristic 103.012347 bit-identical | faithful_unverified (handout ≠ literature; exact DP not exposed to Python) | `m.vendor_managed_inventory_newsvendor_worked_case_summary()`; `m.vendor_managed_inventory_step(...)` |

---

## Group 4 — faithful-but-unverified TRAINABLE env where an adjacent module IS verified (structural gap)

| Problem | Issue | Re-run evidence |
|---|---|---|
| **decentralized_inventory_control / env.rs** | The reusable MDP that heuristics/DP/soft-tree run on does NOT reproduce the 204 anchor: under identical published params it yields **sterman 378 / best base-stock 278** (different pipeline/supply-line bookkeeping). Only the disconnected closed-form port reproduces 204. | `measure_env_vs_closedform.py` re-run: closed-form 204, env.rs 378/278. Only positive env property: Clark-Scarf constant-demand serial optimum = 0.0. |
| **multi_echelon / production_assembly_distribution_network (general network)** | Single-node newsvendor rows reproduce (~0.005 abs), but the general-network / serial protocol reproduces NO published cost; carried Pirhooshyaran serial/network optima are table-only and env does not reproduce 47.65/72.04 under carried echelon levels (documented local-vs-echelon OUL interpretation gap). | `production_assembly_distribution_network_literature_benchmark_summary(...)`: single-node case1 10.6745/12.7111 vs 10.67/12.71. |
| **multi_echelon / assembly** | Structurally anchored by Rosling reduction to serial; literature_verified=false on every instance; no learned-policy env binding, so no trained policy can be evaluated through the normal Python seam. | `scripts/assembly/verify_assembly_rosling_independent.py` passes 3/3 carried instances; `scripts/assembly/benchmark_assembly_policies.py` reproduces the solver-derived assembly optima. |

---

## VERIFICATION DEBT — every snapshot_only_not_rerun (assert literals, do NOT execute the env). Must convert to executing re-run assertions.

| # | Debt | Where | Why it is debt | Fix |
|---|---|---|---|---|
| D1 | **divergent A3C relative rows** (8.95% / 12.09% savings) carried as snapshot literals | multi_echelon/divergent_special_delivery references.rs; `figure_9_gap_labels_are_frozen`-style drift guards | Repo does not implement A3C; the published A3C savings cannot be re-run — only the constant base-stock anchor can. | Either implement an A3C comparator, or relabel these as "published context, not reproduced" and stop implying verification. |
| D2 | **gbk set2 / set3 (+223%)** carried as table-only published rows | multi_echelon/general_backorder_fixed_cost references.rs | The order-per-edge / restricted-transition spec exists only in the gated CEJOR full text; +223% means NOT reproduced. **PARTIALLY ADDRESSED 2026-06-06: both rows now carry `literature_verified=false` + an inline "env does NOT reproduce this cost (+~223%)" comment, so they are no longer presented as verified.** Remaining: recover the transition spec to actually reproduce, or keep as flagged context. |
| D3 | **van Oers 2024 Table 1** two-echelon serial-AM rows are a frozen snapshot | spare_parts_inventory references.rs | No executable two-echelon serial env reproduces them. | Build the two-echelon serial env to re-run, or drop from the benchmark card. |
| D4 | **dual_sourcing l_r=3,4 rows** (#[ignore]d, minutes-scale) | dual_sourcing verification/tests.rs | Validated 2026-06-04 via batch script but NOT re-run this audit; the only fast executing reproduction is the two l_r=2 rows. | Record a dated re-run artifact (the batch script output) and wire an on-demand executing check; treat as faithful+externally-validated until then. |
| D5 (latent) | The **`figure_9_gap_labels_are_frozen` / drift-guard** tests in dual_sourcing assert carried==published literals and do NOT run the env | dual_sourcing verification/tests.rs | A frozen snapshot is NOT verification (per repo MEMORY "verification bar = executing assertion"). The real verification is the executing l_r=2 tests; ensure no one reads the drift guard as the verification. | Keep the drift guard but label it clearly as a drift guard, not a verification; ensure the executing test is the canonical one. |

~~Also note (not snapshot, but missing re-run): ameliorating_inventory perfect-info LP bound has no Python binding~~ **DEBT CLOSED 2026-06-06.** The binding `ameliorating_inventory_perfect_info_lp_bound_summary(reference_name)` now re-runs the perfect-info LP in <1 s and reproduces the companion (Pahr–Grunow 2025 companion code) bound to ~1e-8 (spirits_0001 1991.9344293931 vs 1991.9344293377; port_wine 2444.801). The LP **bound** is now verified_rerun against the companion code; the **trainable env** stays faithful_unverified (the LP is an upper bound, not a published achieved cost the env reproduces).
