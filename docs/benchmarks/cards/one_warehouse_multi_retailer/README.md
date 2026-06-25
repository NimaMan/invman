# Benchmark card — `one_warehouse_multi_retailer`

**Subfamily:** OWMR (Kaynov et al. 2024), 3 regimes: lost_sales / backorder / partial_backorder

**Difficulty:** `hard` — High-dim state/action: 1 warehouse + up to K=10 retailers with a joint allocation decision (proportional / min_shortage / random_sequential) on top of ordering. No exact optimum for the full 100-period instances (only a reduced 2-retailer self-consistency DP anchor 8.485); only 2 of 14 published rows tightly re-run, comparator is a grid-searched gate + cross-protocol PPO context.

**Verification tier:** `faithful` (faithful_unverified (validated only vs the repo's own exact DP))

> Status (manifest, verbatim): verified_rerun (2 of 14 published rows + repo-native exact-DP anchor); remaining 12 rows reproduce ~1-6% off and are carried as table literals

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| kaynov2024_instance_1 | true (ROW PROVENANCE only, NOT tight numerical reproduction) | regime:backorder, K:3, symmetric, demand:Poisson(3), leadtime:Lw2_Lr1, cv:low |
| kaynov2024_instance_2 | true (provenance) | regime:backorder, K:3, heterogeneous, demand:Uniform(0,6)/Normal(3,1)/Poisson(3), leadtime:Lw2_Lr1 |
| kaynov2024_instance_3 | true (provenance) | regime:backorder, K:3, heterogeneous, cv:high, leadtime:Lw2_Lr1 |
| kaynov2024_instance_4 | true (provenance) | regime:backorder, K:3, demand:Poisson(3), leadtime:Lw2_Lr_asymmetric_1_2_3 |
| kaynov2024_instance_5 | true (provenance) | regime:backorder, K:3, cv:high, leadtime:Lw5_Lr3 |
| kaynov2024_instance_6 | true (provenance) | regime:lost_sales, K:3, symmetric, demand:Poisson(3), leadtime:Lw1_Lr1, cv:low |
| kaynov2024_instance_7 | true (provenance; reproduced by re-run -0.94%) | regime:lost_sales, K:3, symmetric, demand:Poisson(3), leadtime:Lw2_Lr1, primary_reference_instance |
| kaynov2024_instance_8 | true (provenance) | regime:lost_sales, K:3, demand:Poisson(3), leadtime:Lw5_Lr1 |
| kaynov2024_instance_9 | true (provenance) | regime:lost_sales, K:3, demand:Poisson(3), leadtime:Lw2_Lr_asymmetric_1_2_3 |
| kaynov2024_instance_10 | true (provenance) | regime:lost_sales, K:3, cv:high, leadtime:Lw5_Lr3 |
| kaynov2024_instance_11 | true (provenance; reproduced by re-run +0.13%) | regime:partial_backorder, K:3, symmetric, demand:Poisson(3), leadtime:Lw2_Lr1, emergency_prob:0.8 |
| kaynov2024_instance_12 | true (provenance) | regime:partial_backorder, K:3, heterogeneous, cv:high, emergency_prob:0.8, learned_gate_beat_target |
| kaynov2024_instance_13 | true (provenance) | regime:partial_backorder, K:10, symmetric_high_cv, demand:Normal(5,14), leadtime:Lw2_Lr2, high_penalty:60 |
| kaynov2024_instance_14 | true (provenance) | regime:partial_backorder, K:10, strongly_heterogeneous, leadtime:Lw2_Lr2, high_penalty:60 |

## Baselines

**Heuristics**
- echelon_base_stock + proportional allocation (Kaynov Eq.8 floor)
- echelon_base_stock + min_shortage allocation
- echelon_base_stock + random_sequential allocation
- grid-searched echelon base-stock 'gate' = better of {proportional, min_shortage}, re-scored on disjoint CRN block

**Exact solver / bound**

Reduced finite-horizon DP on VERIFICATION_PROBLEM_INSTANCE (2 retailers, binary demand, 2-period). Optimal=8.485 dominates both heuristics (9.2225). Repo-native anchor (literature_verified=false), NOT a published number; no exact solver for full 100-period Kaynov instances.

**Published rows**
- instance_7 (lost_sales): proportional -1406.27, min_shortage -1408.08, PPO -1405.08 (gap -0.09%)
- instance_11 (partial_backorder): proportional -1111.76, min_shortage -1109.96, PPO -971.86 (gap -12.58%)
- instance_12: proportional -1402.38, min_shortage -1406.43, PPO -1118.92 (-20.21%)
- instance_13 (K=10): proportional -101727.47, PPO -79727.39 (-21.63%)
- instance_14 (K=10): proportional -53358.86, PPO -42835.02 (-19.72%)
- Full PDF NOT byte-verified (Cloudflare bot wall)

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | False | no | Symmetric Poisson(3) K=3 (1/6/11/7): learned TIES grid-searched gate (0.0000%); warm-started constant leaf reproduces gate at gen 0. No win claimed. |
| `multi_seed_mean_std` | False | yes | instance_12: learned per-retailer echelon_targets_with_alloc_targets linear-leaf beats tuned gate by +4.63% (1115.44 ± 5.51 vs 1169.59) over 6 optimizer seeds; all seeds beat gate. |
| `multi_seed_mean_std` | True | no | instance_12 vs published PPO: learned closes gap to within seed noise (mean +0.31%, 5/6 seeds below PPO); seeds STRADDLE PPO line. No robust PPO beat. |
| `multi_seed_mean_std` | False | yes | instance_13 (K=10 high-CV): linear-leaf beats gate by +7.16% (85310 ± 946 vs 91890.25) over 6 optimizer seeds; all seeds beat gate. |
| `single_seed` | False | no | instance_14 (K=10 strongly heterogeneous): learned TIES strong gate (50445.20, +0.00%). Search-limited. No win. |
| `single_seed` | False | no | Backorder/lost-sales K=3 (3/9/10): learned TIES gate exactly (+0.00%) on all three; below published PPO (no PPO claim). |

## How to reproduce & compare

**Expected (published) value:** instance_7 lost_sales min_shortage published cost 1408.08; instance_11 partial_backorder proportional 1111.76. Plus repo-native exact-DP anchor optimal=8.485 (NOT published).

**Reproduced value (this audit):** instance_7 min_shortage (S_w=44, S_r=[10,10,10], seed 2222, 100 periods x 1000 reps) -> 1394.82, gap -0.94% (tol 1.2%). instance_11 proportional (S_w=43, S_r=[6,6,6]) -> 1113.17, gap +0.13% (tol 0.5%). Exact-DP -> optimal 8.485, both heuristics 9.2225.

**Rerun method / tolerance:** ir.one_warehouse_multi_retailer_exact_dp_summary(); ir.one_warehouse_multi_retailer_get_reference_instance(name) + ir.one_warehouse_multi_retailer_simulate_policy('echelon_base_stock',[Sw,*Sr],...,periods=100,replications=1000,seed=2222,discount=1.0,allocation). Mean-filled warm start.

**Reproduce command(s):**

```bash
python -c "import invman_rust as ir; print(ir.one_warehouse_multi_retailer_exact_dp_summary())"
# instance_7 min_shortage + instance_11 proportional via one_warehouse_multi_retailer_simulate_policy, periods=100, replications=1000, seed=2222, discount=1.0, mean-filled warm start (see verification.rerun_method)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py
for s in 841 842 844 845; do RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets_with_alloc_targets --policy_state_mode absolute_augmented --leaf_type linear --depth 3 --split_type axis_aligned --temperature 0.10 --warm_start_at_best_base_stock --sigma_init 0.05 --gate_search_paths 64 --training_episodes 800 --es_population 32 --train_seed_batch 24 --holdout_paths 4096 --train_allocation min_shortage --same_seed --seed $s; done
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets --leaf_type linear --warm_start_at_best_base_stock --train_allocation proportional --same_seed
python scripts/one_warehouse_multi_retailer/verify_ppo_beat_disjoint_blocks.py
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
