# invman benchmark — index

A **library of inventory-control systems** for learning and benchmarking ordering policies. Each system ships with:

- **reference instances** — named, parameterized problem instances with `coverage_dimensions` tags (regime / demand / lead-time / CV / K / penalty …), catalogued in `BENCHMARK_MANIFEST.json`;
- **baselines** — heuristic policies (myopic, base-stock, (s,S), capped-dual-index, anchor-and-adjust, …) and, where a tractable solver exists, an **exact** optimum or **bound** (value iteration, Clark–Scarf, bounded-DP, perfect-information LP);
- **reproducible results** — a learned soft-tree / FlowNet policy compared against the strongest baseline ("gate") under a pinned seed protocol, reported (where finalized) as **mean±std over ≥5 optimizer seeds**.

**Verification is layered and labeled honestly.** A result is one of:
1. **verified_rerun (peer-reviewed)** — a published *peer-reviewed* number reproduced by *actually re-running* the env/solver this audit;
2. **verified_rerun (reference-impl / companion / closed-form)** — reproduced by re-run, but the source is author companion code, an instructor handout, or a closed-form port — *not* a peer-reviewed article table;
3. **faithful_unverified** — the env is structurally faithful but only a **repo-native self-consistency anchor** (an exact DP the repo itself defines) re-ran, or **no public per-instance number exists** to target.

A frozen snapshot (carried `literal == published` constant, env *not* executed) is **never** verification — those are tracked as **debts** below.

---

## Baseline API — get the baselines for a problem with minimal effort

`docs/benchmarks/BENCHMARK_MANIFEST.json` is the **single source of truth** (14 problem families). `invman.benchmarks.catalog` is a dependency-light (stdlib-only) API **over** that manifest — it never duplicates the data, it loads + structures it. Use it to pull a problem's reference instances, baselines, reference results, difficulty, verification tier, and exact reproduce commands in a few lines:

```python
from invman.benchmarks import catalog

catalog.list_problems()                    # -> 14 problem names, in manifest order
catalog.list_problems(difficulty='easy')   # filter by difficulty: easy|medium|hard
catalog.list_problems(verified='strict')   # filter by verification tier: strict|reference|faithful

card = catalog.get('lost_sales')           # -> ProblemCard (accepts short name or full manifest 'problem' string; raises KeyError on unknown)
card.difficulty                            # 'easy'
card.verification.tier                     # 'strict' | 'reference' | 'faithful'
card.instances                             # [Instance(name, dimensions, literature_verified_flag), ...]
card.baselines.heuristics                  # list[str]; also .exact_solver, .published_rows
card.results                               # [Result(claim, seed_reporting, at_risk), ...]
card.reproduce_commands                    # list[str] — exact commands to regenerate the baseline

print(catalog.render_card('lost_sales'))   # -> the Markdown BENCHMARK_CARD as a string
catalog.render_all_cards('docs/benchmarks/cards')   # writes one card per problem + index
```

The pre-rendered cards live in [`cards/`](./cards/) — one **BENCHMARK_CARD** per problem with instances, baselines, reference results, and a **"How to reproduce & compare"** block (command + expected value + tolerance) so a consumer can regenerate the baseline and compare their own approach. Regenerate them any time with `catalog.render_all_cards('docs/benchmarks/cards')`.

### Executable layer — run a baseline, not just read it

`catalog` reads metadata; `invman.benchmarks.runners` **runs** it. `catalog.get(problem).load_instance(name)` returns a runnable `ReferenceInstance` that carries the env params + published baselines, re-runs those baselines on the live env, and scores your own soft-tree policy on the same instance — through the *same* seam the CMA-ES optimizer uses (no second, drifting evaluator).

```python
from invman.benchmarks import catalog

inst = catalog.get('lost_sales').load_instance('lit_poisson_p4_l4')
inst.published_costs        # {'optimal': 4.73, 'myopic2': 4.82, ...}
inst.run_baselines()        # re-run the shipped baselines on the live env
my_cost = inst.evaluate(my_trained_params)   # score your policy (size it with inst.policy_param_count())
inst.compare(my_cost)       # signed gap vs the reference + a 'beats' verdict
```

**All 14 catalog families have a runner** (157 reference instances) — every one supports `list_instances` / `load_instance` / `published_baselines` / `run_baselines` / `compare`; `lost_sales` (+fixed), `dual_sourcing`, and `multi_echelon` additionally support `evaluate` (their soft-tree rollout is in the CMA-ES seam). See [`../../invman/benchmarks/runners/README.md`](../../invman/benchmarks/runners/README.md) for the per-family table. Worked per-family reports that emit a published-vs-recomputed comparison table live in [`../../scripts/benchmark_baselines/`](../../scripts/benchmark_baselines/) (`run_<family>_baselines.py --simulate`).

### Difficulty rubric (`easy` / `medium` / `hard`)

Each manifest entry carries a `difficulty` plus a one-line `difficulty_rationale`. Difficulty **folds three axes**:

1. **State/action dimensionality** — scalar single-item state + scalar order action is easiest; age-stratified / multi-echelon / multi-retailer / decentralized state + vector or joint (allocation, pricing, blend, ship) actions are hardest.
2. **Exact-solver availability** — a problem with an **exact VI/DP true optimum** (or a tractable reduced verifier) is *easier to benchmark*, because the comparator is a clean denominator; only a bound or a self-consistent anchor is harder.
3. **Comparator type** — `true_optimum_match_only` (easiest to score) < `heuristic_to_beat` ≈ `bound_gap` < `self_consistent` (hardest to score honestly).

The split (and the rationale per problem) is in the manifest; the headline assignment:

- **easy** — `lost_sales`, `joint_pricing_inventory`, `procurement_removal_inventory`, `random_yield_inventory`, `spare_parts_inventory` (low-dim single-item, exact DP/VI true optimum, clean heuristic/optimum comparator).
- **medium** — `dual_sourcing`, `perishable_inventory`, `ameliorating_inventory`, `joint_replenishment`, `nonstationary_lot_sizing`, `vendor_managed_inventory` (coupled actions, forecast/age state, or a bound-only / proxy comparator).
- **hard** — `multi_echelon`, `one_warehouse_multi_retailer`, `decentralized_inventory_control` (high-dim networks / allocation / decentralized info; no exact optimum for the full instance, mostly self-consistent or bound-gap comparators). `multi_echelon` is an umbrella entry — its per-subfamily difficulty (serial=medium, rest=hard) is recorded in `difficulty_by_subfamily` in the manifest.

### Verification tiers (the honest label the API derives)

`card.verification.tier` is **derived** from the manifest `verification.status` string by its strongest verified component, and is exactly the layered honesty model above:

- **`strict`** — `verified_rerun` against a **peer-reviewed** published number (README Group 1).
- **`reference`** — `verified_rerun` against a **reference-impl / companion-code / closed-form** number (not a peer-reviewed article table; README Group 2: `nonstationary_lot_sizing`, `decentralized_inventory_control`).
- **`faithful`** — `faithful_unverified`: env faithful but only a repo-native self-consistency anchor re-ran, or no public per-instance number exists (README Group 3).

Note the tier is the *strongest verified component* of a possibly-mixed status — e.g. `spare_parts_inventory` is `strict` via the Kranenburg analytical module even though its *trainable env* is `faithful_unverified` (caveat in its card), and `joint_replenishment` is reported `strict` because its published quantity is an action `q=(0,6)` re-derived by VI even though no published *cost* table exists. Always read the card's full `Status (manifest)` line for the nuance — the master table below is the editorial Group 1/2/3 partition.

## How it maps to the paper

The companion paper is `paper/learning_inventory_control_policies_es.tex` (compiled `paper/learning_inventory_control_policies_es.pdf`). Most systems have a dedicated `\section`; several systems in this benchmark library are **benchmark-only** (no paper section) or appear only as **related-work / future-work context**. The "paper §" column below is the honest mapping — `—` means not written up in the paper.

---

## Master table — one row per system (multi_echelon split by sub-family)

Grouped by verification status (best provenance first), matching `VERIFICATION_LEDGER.md`.

### Group 1 — verified_rerun against a PEER-REVIEWED published number

| System | #inst | Heuristic? | Exact/Bound? | Verification | Paper § | Card |
|---|---|---|---|---|---|---|
| **lost_sales** | 7 | ✓ | ✓ exact VI (fixed-cost) | verified_rerun — Bijvank 2015 Table 1 (fixed-cost, gaps <0.005) + Zipkin 2008 Table 3a (3 vanilla heuristic rows ~0.015); vanilla optimum 4.73 is a *carried* Zipkin DP value | §"Lost sales and fixed-cost lost sales" | [card](../../src/problems/lost_sales/BENCHMARK.md) |
| **dual_sourcing** | 6 | ✓ | ✓ bounded DP (proxy, not proof-optimum) | verified_rerun — Gijsbrechts 2022 Fig 9 gaps, two l_r=2 rows re-run (≤0.0075pp); l_r=3,4 = debt D4 | §"Dual sourcing" (`sec:dual-sourcing`) | [card](../../src/problems/dual_sourcing/BENCHMARK.md) |
| **perishable_inventory** | 6 | ✓ | ✓ exact VI (≤2000 states) | verified_rerun — genuine VI re-derivation of 3 independent quantities (De Moor 2022 tables + best base-stock; Farrington 2025 Table 3 VI −1457/−1553) on the four m=2/L=1 instances; 28 Scenario-A rows table-only | §Perishable inventory (`sec:perishable`) | [card](../../src/problems/perishable_inventory/BENCHMARK.md) |
| **multi_echelon / serial** | 3 | ✓ | ✓ exact Clark–Scarf | verified_rerun — Snyder & Shen Ex 6.1 = 47.65 → 47.6654; Poisson/multi-stage vs stockpyl | §`sec:serial` | [card](../../src/problems/multi_echelon/serial/BENCHMARK.md) |
| **multi_echelon / general_backorder_fixed_cost** | 3 | ✓ | — (audit base-stock) | verified_rerun (set1 10467→10384.9; Kunnumkal-Topaloglu 4059→3933.3); set2/set3 = debt D2 (+223%, NOT reproduced) | §`sec:genbackorder` | [card](../../src/problems/multi_echelon/general_backorder_fixed_cost/BENCHMARK.md) |
| **spare_parts_inventory** ⚠ adjacent module only | 4 | ✓ | ✓ Kranenburg analytical + repo DP | split: Kranenburg 2006 Table 5.2 (35/35 rows ≤0.02) is verified_rerun but a **structurally different** continuous-review model — does NOT verify the trainable env (which is faithful_unverified, see Group 3); van Oers 2024 = debt D3 | — (not covered) | [card](../../src/problems/spare_parts_inventory/BENCHMARK.md) |

### Group 2 — verified_rerun against a reference-impl / companion / closed-form number (NOT a peer-reviewed article table)

| System | #inst | Heuristic? | Exact/Bound? | Verification | Paper § | Card |
|---|---|---|---|---|---|---|
| **nonstationary_lot_sizing** | 11 | ✓ | — (rolling-DP is strongest baseline) | verified_rerun vs **author companion-code CSVs** (HenriDeh/DRL_MMULS), not an EJOR table: constant_10 simple (s,S) 1834.918 (+0.109%), rolling-DP 1714.148 (+0.141%); (s,S) levels & DP first-period EXACT. `literature_verified=false` (honest) | — (no dedicated section) | [card](../../src/problems/nonstationary_lot_sizing/BENCHMARK.md) |
| **joint_replenishment** | 3 | ✓ | ✓ VI (action) + finite-horizon DP | verified_rerun of a published **ACTION** q=(0,6) (Vanvuchelen 2020 Fig 3), re-derived by VI; **no published cost table exists** (costs are repo-native) | §related-work / future-work only | [card](../../src/problems/joint_replenishment/BENCHMARK.md) |
| **decentralized_inventory_control** | 3 | ✓ | ✓ closed-form port + reduced DP (test-only) | verified_rerun of closed-form board-game total **204** (EXACT); the trainable `env.rs` MDP yields 378/278 under identical params and does NOT reproduce 204 (faithful_unverified, see Group 4) | — (not covered) | [card](../../src/problems/decentralized_inventory_control/BENCHMARK.md) |

### Group 3 — faithful_unverified (env faithful; only repo-native self-consistency re-ran, or no public number to target)

| System | #inst | Heuristic? | Exact/Bound? | Verification | Paper § | Card |
|---|---|---|---|---|---|---|
| **ameliorating_inventory** | 4 | ✓ | ✓ perfect-info LP (UPPER BOUND) | faithful_unverified — LP bound NOT re-run this audit (no Python binding; in-crate re-solve test is genuine but read-only); learned env path WAS re-run. README says "literature-verified: TRUE" which **contradicts** the ledger | §"Ameliorating inventory" | [card](../../src/problems/ameliorating_inventory/BENCHMARK.md) |
| **joint_pricing_inventory** | 2 | ✓ | ✓ exact DP (T=5 verifier) | faithful_unverified / no_published_number — analytical critical-fractile y*=(3,2,2) and exact DP optimal −33.178121 (action (2,1)) reproduced (incl. independent Python DP to 1e-9), but neither is a PUBLISHED number | — (not covered) | [card](../../src/problems/joint_pricing_inventory/BENCHMARK.md) |
| **procurement_removal_inventory** | 3 | ✓ | ✓ exact DP (reduced verifier only) | faithful_unverified / no_published_number — exact-DP optimal 31.78026 to 1e-10 + interval_stock rows reproduced exactly; both cited papers are CONTEXT (2017 NPV ~84000; 2025 qualitative) | — (related-work only) | [card](../../src/problems/procurement_removal_inventory/BENCHMARK.md) |
| **random_yield_inventory** | 5 | ✓ | ✓ exact reduced DP (single slice) | faithful_unverified / no_published_number — repo-native exact-DP anchor 40.0599 reproduced; Yan 2026 / Chen 2018 paywalled, Inderfurth 2015 is a different yield model | — (benchmark-only) | [card](../../src/problems/random_yield_inventory/BENCHMARK.md) |
| **vendor_managed_inventory** | 8 | ✓ | ✓ finite-horizon DP (Rust-only, no Python binding) | faithful_unverified / no_published_number — paper table paywalled; only OPEN numbers are an instructor **handout** (MDH 15 / six-sigma 31.53 / newsvendor 26.96), reproduced exactly but a handout is NOT literature verification; optimality ceiling not exposed | — (not covered) | [card](../../src/problems/vendor_managed_inventory/BENCHMARK.md) |
| **multi_echelon / assembly** | 1 (3 sub) | ✓ | ✓ Rosling reduction → serial solver | faithful_unverified — verified-by-equivalence only (Rosling reduction); all instances `literature_verified=false`; NOT re-run via bindings this audit (manual remap gave 26.55 ≠ 22.759) | — (no dedicated section) | [card](../../src/problems/multi_echelon/assembly/BENCHMARK.md) |
| **multi_echelon / divergent_special_delivery** | 4 | ✓ | — (best constant base-stock by grid) | verified_rerun on Van Roy const base-stock (51.7/1302/1449 → within 2%); A3C savings rows = debt D1 (repo implements no A3C) | §`sec:multiechelon` | [card](../../src/problems/multi_echelon/divergent_special_delivery/BENCHMARK.md) |
| **multi_echelon / production_assembly_distribution_network** | 2 | ✓ | analytical single-node only | verified_rerun on single-node analytical rows (~0.005 abs) + faithful_unverified on general-network/serial/mixed/pure-assembly (no published cost reproduced; local-vs-echelon OUL gap); adjacent van Oers 2024 = debt D3 | §`sec:pirhoo` | [card](../../src/problems/multi_echelon/production_assembly_distribution_network/BENCHMARK.md) |

### Cross-cutting

| System | #inst | Heuristic? | Exact/Bound? | Verification | Paper § | Card |
|---|---|---|---|---|---|---|
| **one_warehouse_multi_retailer** | 14 | ✓ | ✓ reduced exact DP (self-consistency) | verified_rerun on 2 of 14 Kaynov 2024 rows (instance_7 −0.94%, instance_11 +0.13%) + repo-native exact-DP anchor 8.485; remaining 12 rows reproduce ~1–6% off (carried table literals); full PDF not byte-verified | §`sec:owmr` | [card](../../src/problems/one_warehouse_multi_retailer/BENCHMARK.md) |
| **multi_echelon** (parent) | 13 | — | — | umbrella over the 5 sub-families above; verdicts differ per sub-family | §`sec:serial`/`sec:multiechelon`/`sec:pirhoo`/`sec:genbackorder` | [card](../../src/problems/multi_echelon/BENCHMARK.md) |

> Instance counts are manifest entries (`BENCHMARK_MANIFEST.json`); some single entries cover multiple sub-instances (e.g. assembly = 1 entry / 3 components; padn single-node = 1 entry / 7 cases). Multi_echelon sub-family split: serial 3, assembly 1(×3), divergent 4, padn 2, gbk 3 = 13.

---

## Backbone artifacts

- **`BENCHMARK_MANIFEST.json`** — the machine-readable spine: per-system instances + `coverage_dimensions`, baselines (heuristic / exact), per-result `seed_reporting` + `at_risk` flags, and `reproduce_commands`. ([file](./BENCHMARK_MANIFEST.json))
- **`VERIFICATION_LEDGER.md`** — the honest per-system verdict: what published number was reproduced **by re-run** this audit, by what method, and the four standing snapshot debts (D1–D5). ([file](./VERIFICATION_LEDGER.md))
- **`PROPER_REPO_BUILD_PLAN.md`** — the work plan to turn scattered logs into a proper benchmark repo: standard API per problem, verification-debt closures (V1–V8), seed-robustness debts (S-H/S-M/S-L), reproducibility spine (R1–R5), cleanup (F1–F5). ([file](./PROPER_REPO_BUILD_PLAN.md))

---

## How to reproduce a result

1. **Open the system's card** (the `Card` link above). Each card carries a **reproduce block**: the exact Python/binding call, the optimizer seed(s), eval/CRN seeds, the expected value, and the tolerance.
2. **Run the named command** against the named instance. The verified anchors above are the ones to start with — they re-run in seconds-to-minutes (e.g. dual_sourcing l_r=2 ≈ 10s; lost_sales fixed-cost exact DP; perishable VI on m=2/L=1).
3. **Compare to the manifest** — `BENCHMARK_MANIFEST.json` carries the same `reproduce_commands` and the `published_value` / `reproduced_value` strings the ledger audited.

### Seed-robust reporting standard (project mandate)

- **Report mean±std over ≥5 *optimizer* seeds.** Never report a single seed or best-of-N as a headline. A large eval-SEM is demand-path CRN noise, **not** optimizer-seed variance — the two must not be conflated.
- A result is **seed-robust** only when the manifest marks it `seed_reporting=multi_seed_mean_std`, `at_risk=false`. Everything marked `single_seed` / `best_of_n` / `at_risk=true` is labeled **"single-seed / NOT yet seed-robust"** in its card and **must not** be read as a robust win. (Per-system reference runner pattern: `seed_robust_<problem>.py`, modeled on `seed_robust_mixed_distribution_assembly_network.py`.)
- **Cross-protocol comparators are context, never a "beats" claim.** Published deep-RL numbers (A3C, PPO, DQN/DDQN, SAA) trained under a *different* protocol/MDP are carried as context only. No card claims a beat over a cross-protocol comparator.
- **Exact solvers / bounds are denominators or sanity ceilings**, not beat targets — including repo-native self-consistency DPs (which are correctness checks, not literature) and perfect-information LP **upper bounds** (gap-to-bound, never an achievable optimum).

---

## Honesty / open debts

Aggregated from the per-card audits, the verification ledger, and the build plan. These are the things a benchmark consumer must not misread.

### Verification debts (snapshot_only → must become executing re-runs)

- **D1 — divergent A3C savings rows (8.95% / 12.09%)**: carried as snapshot literals; the repo implements no A3C, so they cannot be re-run. Drift-guard tests assert literals and are **NOT verification**. Fix: implement A3C or relabel as "published context, not reproduced".
- **D2 — gbk set2/set3 (published 4797)**: env yields ~15497 = **+223%, NOT reproduced**; `literature_verified=false`. Must be flagged NOT-reproduced beside the verified set1/KT rows.
- **D3 — van Oers 2024 Table 1** (two-echelon serial-AM, padn / spare_parts adjacent): frozen snapshot, no executable env reproduces it.
- **D4 — dual_sourcing l_r=3,4**: prior-date validated (2026-06-04) but **NOT re-run this audit**; only the two l_r=2 rows were freshly re-run. `#[ignore]`d, minutes-scale.
- **D5 (latent) — dual_sourcing drift guards** (`figure_9_gap_labels_are_frozen`): assert carried==published literals without executing the env. A frozen snapshot is NOT verification; the executing l_r=2 test is the canonical one.
- **Provenance ≠ numerical reproduction (OWMR / lost_sales)**: an instance-level `literature_verified` flag means **row provenance**, NOT tight numerical reproduction. OWMR: only 2/14 rows tightly re-run; 12 carried at ~1–6% (regime-dependent sign); full Kaynov PDF never byte-verified. lost_sales: the vanilla optimum 4.73 (and 8.89/10.61/22.95) is a carried Zipkin DP value, NOT recomputed in-repo; only the canonical L4-Poisson row carries true Zipkin numbers; the fixed-cost 80-instance grid beyond the Bijvank anchor is not literature-verified; MMPP2 rows are repo-computed.
- **Repo-native anchors are self-consistency, not literature**: OWMR exact-DP 8.485, joint_pricing −33.178, procurement_removal 31.780, random_yield 40.0599, spare_parts 28.39366, decentralized closed-form 204 (the value the R-port emits, NOT a transcribed table line). All `literature_verified=false` and must not be read as published optima.
- **No public per-instance number exists** for joint_pricing_inventory, procurement_removal_inventory, random_yield_inventory, vendor_managed_inventory (paywalled / different model / pricing-coupled NPV); their reproduced numbers are repo-internal anchors or an instructor handout only.
- **Adjacent / structurally-different modules do not verify the trainable env**: spare_parts Kranenburg (continuous-review lateral-transshipment) ≠ the periodic-review trainable env; decentralized closed-form board-game ≠ the trainable `env.rs` (378/278); padn single-node analytical ≠ the general-network protocol.
- **Optimality ceilings unavailable / unexposed**: vendor_managed_inventory and decentralized finite-horizon DP optima are Rust-only (no Python binding) → reported gaps are learned-vs-heuristic, not learned-vs-optimum. dual_sourcing bounded DP is a truncated-box proxy (for l_r=4 it sits ~0.2% below the heuristics, unusable as a denominator → capped-dual-index is the optimal proxy). ameliorating perfect-info LP is an UPPER BOUND (gap-to-bound, never a beat).

### Seed-robustness debts (at_risk headline results → need ≥5 optimizer seeds)

Almost every "beats heuristic/gate" headline is currently a **single optimizer seed or best-of-N** and is labeled NOT-yet-seed-robust in its card:

- **multi_echelon/divergent** settings 1&2 −14.4% (best-of-N); **gbk** set1 −22.4%/−26.7% + Kunnumkal-Topaloglu ~−37% (best-of-N, N=2); **padn** serial case3 / pure-assembly (single-seed env-own-heuristic beats). The padn **mixed** row was CORRECTED from a best-of-3 −0.99% to a seed-robust gate-match (8 seeds 306.10±22.89).
- **lost_sales** vanilla 22/24-instance sweep + fixed-cost 48-instance sweep (single-seed); only the canonical vanilla L4-Poisson Tree-2 = 4.7537 is multi-seed.
- **dual_sourcing** "beats CDI on 2 rows" + factor-screen negatives (single-seed; margins inside CDI's own ≤0.11% band, economically negligible); the only seed-robust result is "matches CDI on all 6".
- **OWMR** instance_13 (+6.44%/+8.27% paper-table form) and instance_12 are reconciled to finalized 6-seed numbers in `SEED_ROBUST_BENCHMARK_2026_06_06.md` (instance_13 +7.16% / 85310±946; instance_12 +4.63% / 1115.44±5.51), but the **paper table still lags** these.
- **perishable** 5 "beats gate" rows (+0.70..+2.21%) single-seed; **ameliorating** spirits/port_wine (+450/+278/+524%) single-seed; **joint_replenishment** 6/16 MOQ-beats single-seed + setting-10 flip best-of-N=2; **joint_pricing** +25.15% single-seed; **nonstationary_lot_sizing** beats DP 8/8 single seed=1234; **random_yield** seed slice is 4 seeds (one short of ≥5) and the d3 headline contradicts a saved d1-linear artifact; **vendor_managed_inventory** low_penalty −0.31% two-seed best-of (marginal tie); **spare_parts** +1.34% single-seed (margin ≪ cost std); **procurement_removal** never beats its gate anyway.

### Documentation / consistency caveats

- **ameliorating_inventory README contradicts the ledger** (README lines 10–12 say "literature-verified: TRUE"; ledger says faithful_unverified). Left unedited per instructions; the card's caveats supersede it.
- **nonstationary_lot_sizing README** has mildly contradictory framing ("literature-verified" header before correctly stating `literature_verified=false`). Left unedited; noted in card.
- **gbk family name is a misnomer** — there is NO fixed ordering cost (holding + backorder only).
- **Demand-convention fix (OWMR)**: `RoundedNormal` param2 is the standard deviation, not variance (N(5,14) ⇒ mean 5 / std 14 / σ/μ=2.8).
- **Dead scripts** noted (perishable `run_paper_benchmark.py`/`common.py`, plus procurement/random_yield scratch scripts) import the removed `invman.policies.soft_tree` and do not run; build-plan item F1 removes them.
- **Reproducibility spine gaps** (build-plan R-items): several headline numbers live only in gitignored `outputs/` + markdown; trained parameter vectors aren't all committed, so some learned rows can't be re-evaluated without retraining. Targets: a committed results table tied to seeds (R1) and one paper-table regenerator (R2).

> All existing per-folder `README.md` files that were *consistent* with the code/manifest were left untouched; the two contradictory READMEs (ameliorating, nonstationary) were left unedited per the no-overwrite instruction and the contradictions surfaced in their cards and here.
