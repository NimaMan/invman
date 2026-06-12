# Benchmark card — `spare_parts_inventory`

**Subfamily:** single-echelon periodic-review repairable (trainable env) + Kranenburg 2006 continuous-review lateral-transshipment (verification-only)

**Difficulty:** `easy` — Trainable env is a low-dim single-echelon repairable item (on-hand + repair/procurement pipelines) with a scalar order-up-to action; the reduced verifier has an EXACT bounded DP true optimum (28.39366, true_optimum_match_only) and a base-stock comparator. (The Kranenburg analytical module that IS literature-verified is a structurally different continuous-review model, not the benchmark env.)

**Verification tier:** `reference` (re-runs a companion / closed-form / reduced-module number, or a published action)

**Tier note:** Split: the strict bit is the Kranenburg 2006 ANALYTICAL lateral-transshipment module (35/35 Table 5.2 rows re-run) — a DIFFERENT continuous-review model. The benchmarked TRAINABLE periodic-review repairable env is faithful_unverified (no published cost for that exact construction). Headline tier = reference because the only literature anchor is the adjacent analytical module, not the env.

> Status (manifest, verbatim): verified_rerun (ONLY for the Kranenburg analytical lateral-transshipment module, which is structurally DIFFERENT from the trainable env); trainable env = faithful_unverified; van_oers table = snapshot_only_not_rerun

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| single_echelon_repairable_operational_spares (PRIMARY) | false | regime:backorder, review:periodic, repairable:deterministic_repair_return, horizon:17_periods, installed_base:12, L_proc:3, L_repair:2, p_fail:0.08, downtime_cost:20, trainable_env |
| VERIFICATION_PROBLEM_INSTANCE (reduced exact-DP) | false | regime:backorder, review:periodic, repairable, horizon:4_periods, discount:0.99, installed_base:3, L_proc:2, L_repair:2, p_fail:0.4, max_order:4, exact_dp_tractable |
| kranenburg2006_table5_2 (35 rows) | true | model:continuous_review_METRIC, multi_location, lateral_transshipment, emergency_replenishment, analytical_exact, different_model_from_env |
| van_oers2024_table1 (no_am / upstream_am / downstream_am) | false | echelons:2, review:periodic, serial, additive_manufacturing, recorded_table_only, frozen_snapshot |

## Baselines

**Heuristics**
- base_stock (order-up-to S; benchmark S=5, best-constant S=6)
- lead_time_mean_cover (safety_buffer over expected lead-time failures; buffer=1.0)

**Exact solver / bound**

finite_horizon_dp.rs solve_optimal_policy — bounded backward-induction DP, tractable only on the reduced VERIFICATION_PROBLEM_INSTANCE; self-consistency comparator, NOT a published optimum. Plus the Kranenburg analytical exact solver (R* enumeration) for the SEPARATE lateral-transshipment sub-family.

**Published rows**
- Kranenburg 2006 Table 5.2 base case: Situation1 R*=9.09 C=91.90; Situation3 R*=6.10 C=63.00; ratio 1.46 (all 35 rows re-run and matched within 0.02)
- van Oers 2024 Table 1 no-AM: enumeration 100.0/99.57; newsvendor 117.0/99.08; echelon_separation 105.9/99.36 (RECORDED ONLY, frozen, not re-run)
- van Oers 2024 downstream-AM: enumeration 71.98; echelon_separation 72.01 (recorded only)

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `best_of_n` | True | no | Learned soft-tree (depth2, oblique, linear, T=0.10) beats best constant base-stock S=6 by 1.34% out-of-sample (53.06 vs 53.78) on 4096-seed holdout; beats S=5 by 15.77% and lead_time_mean_cover by 42.92%. |
| `single_seed` | True | no | Re-run reproduction: soft_tree=50.72 vs best-constant S=6=54.44 = 6.84% on 512-seed holdout; consolidated 4096-seed JSON reproduced 53.06 vs 53.78 = 1.34% exactly. |
| `none` | False | no | Repo-native exact DP weakly dominates both heuristics on reduced verifier (optimal 28.394 <= base_stock 28.394 <= lead_time_mean_cover 28.912). |

## How to reproduce & compare

**Expected (published) value:** Kranenburg 2006 Table 5.2 base case (m=0.001): Situation1 R1*=9.09 C1=91.90; Situation3 R3*=6.10 C3=63.00; ratio 1.46 (35 rows, TU/e thesis Ch.5 p.107)

**Reproduced value (this audit):** All 35/35 rows within table-rounding tolerance 0.02; worst abs diff 0.005. Base case R1*=9.0900 C1=91.9000, R3*=6.1000 C3=63.0000, ratio=1.4587. Repo-native exact DP: optimal=28.39366, base_stock=28.39366 (gap 0.0), lead_time_mean_cover=28.91225 (gap 0.519) — DP weakly dominates both but matches NO published number.

**Rerun method / tolerance:** looped spare_parts_inventory_kranenburg_reference_instances() -> spare_parts_inventory_kranenburg_exact_summary(name) over all 35 rows checking all_within_tolerance; and spare_parts_inventory_exact_dp_summary(). Both <2s.

**Reproduce command(s):**

```bash
python -c "import invman_rust as m; rows=m.spare_parts_inventory_kranenburg_reference_instances(); n=sum(m.spare_parts_inventory_kranenburg_exact_summary(r['name'])['published_table_comparison']['all_within_tolerance'] for r in rows); print(n, '/', len(rows))"
python -c "import invman_rust as m; s=m.spare_parts_inventory_exact_dp_summary(); print(s['optimal_discounted_cost'], s['base_stock_gap_to_optimal'], s['lead_time_mean_cover_gap_to_optimal'])"
python scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py --holdout_seeds 4096 --holdout_seed_start 900000
python scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py --holdout_seeds 512 --holdout_seed_start 900000
python scripts/spare_parts_inventory/train_soft_tree_reference.py --seed 123 --depth 2 --temperature 0.10
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._

