# Benchmark card — `joint_replenishment`

**Subfamily:** Vanvuchelen, Gijsbrechts & Boute 2020 FTL two-item

**Difficulty:** `medium` — Two coupled items sharing a truck (joint setup + capacity constraint) makes the action a 2-vector with a shared-fixed-cost interaction; a setting-5 VI reproduces the published optimal ACTION q=(0,6) but there is NO published absolute-cost table, so costs are repo-native (self-consistent) and the comparator is heuristic_to_beat (MOQ).

**Verification tier:** `reference` (re-runs a companion / closed-form / reduced-module number, or a published action)

**Tier note:** Reference (not strict): VI reproduces the published optimal ACTION q=(0,6) at state (5,0), not a printed cost (no published absolute-cost table exists).

> Status (manifest, verbatim): verified_rerun (published quantity is an ACTION q=(0,6), re-derived by VI; finite-horizon DP self-consistency reproduced; no published cost exists)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| vanvuchelen2020_small_scale_setting_5 (PRIMARY + Figure-3 anchor) | absent (no literature_verified field; carried verbatim from Table 2) | regime:backorder, items:2, truck_capacity:V6, K:75, minor:asymmetric_k[40,10], holding:low_h[1,1], shortage:b[19,19], demand:U[0,5]xU[0,3], gamma:0.99, has_vi_optimum |
| vanvuchelen2020_small_scale_setting_1..16 (16 Table-2 settings) | absent (definitions match Table 2; no per-setting cost reproduced) | regime:backorder, items:2, truck_capacity:V6, K:75, h_in{1,5}, b_in{19,95}, minor_k_in{[10,10],[40,10]}, gamma:0.99 |
| VERIFICATION_PROBLEM_INSTANCE (reduced 4-period DP) | false (repo_finite_horizon_self_consistency_comparator) | regime:backorder, items:2, periods:4, self_consistency_only |

## Baselines

**Heuristics**
- minimum_order_quantity (MOQ / (Q,S|T)) — strongest on all 16 settings
- dynamic_order_up_to (DYN-OUT) — dominated by MOQ

**Exact solver / bound**

TWO: (1) in-crate reduced finite-horizon DP (4-period, self-consistency only); (2) infinite-horizon discounted VI specialised to setting 5 — reproduces published Figure-3 optimal ACTION q=(0,6) at state (5,0). Paper gives NO absolute optimal-cost table.

**Published rows**
- Figure-3 optimal ACTION (setting 5, state (5,0)): q=(0,6) (verified verbatim, re-derived by VI)
- Figure-3 heuristic ACTION: q=(2,4) — STORED LITERAL; repo MOQ orders (0,6), NOT (2,4)
- Figure 2: heuristics 4-25% above optimal (no extractable numbers); repo setting-5 MOQ gap +19.64%
- No published per-setting absolute cost; paper PPO costs not carried

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | Learned soft-tree beats MOQ on 6 of 16 settings: setting 5 +13.05%, 13 +11.44%, 14 +6.45%, 6 +4.23%, 9 +1.07%, 1 +0.51% (single optimizer seed 123) |
| `single_seed` | True | no | Setting 5: learned soft-tree +3.14% above VI optimum (6546.176 vs 6347.108), closing 84% of MOQ's +19.64% gap; beats MOQ -13.79% on all paths |
| `best_of_n` | True | no | Setting 10: autoresearch flips loss to WIN gap -0.85%; 'robust across two seeds' s123 -0.79% / s777 -0.85% |
| `best_of_n` | True | no | Setting 7: closed to +0.09% near-tie (best of 7 seeds); never strictly flips, remains a loss to MOQ |
| `single_seed` | False | no | MOQ dominates DYN-OUT on all 16; learned LOSES on high-cost h=5,b=95 family by -8.9% to -18.1% (policy-class limit) |

## How to reproduce & compare

**Expected (published) value:** Setting-5 Figure-3 optimal action q=(0,6) at state (5,0). No published absolute cost.

**Reproduced value (this audit):** VI greedy action at (5,0) = (0,6), converged iter 2260, max delta 9.92e-09. Finite-horizon DP comparator: optimal (6,6) 266.386, MOQ (7,5) 386.101, DYN-OUT (6,6) 383.960 (matches README literals). Setting-5 VI-optimum mean cost 6347.108 is faithful but JSON/model gitignored, NOT re-run.

**Rerun method / tolerance:** invman_rust.joint_replenishment_exact_dp_summary(); and benchmark_vanvuchelen_settings.value_iteration_setting5(lo=-12,hi=18); g((5,0)). Both <2 min.

**Reproduce command(s):**

```bash
python -c "import invman_rust as ir; print(ir.joint_replenishment_exact_dp_summary())"
cd /home/nima/code/ml/invman && python -c "import sys; sys.path.insert(0,'scripts/joint_replenishment'); import benchmark_vanvuchelen_settings as bvs; g,it,d=bvs.value_iteration_setting5(lo=-12,hi=18); print(tuple(int(x) for x in g((5,0))[0]), it, d)"
python -c "import invman_rust as ir; print(ir.joint_replenishment_published_action_anchor())"
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/benchmark_learned_vs_heuristics.py
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/autoresearch_joint_replenishment.py --budget full --warm_start_moq --reference vanvuchelen2020_small_scale_setting_5 --seed 123
RAYON_NUM_THREADS=2 python scripts/joint_replenishment/evaluate_setting5_vs_vi_optimum.py --eval_paths 4096
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
