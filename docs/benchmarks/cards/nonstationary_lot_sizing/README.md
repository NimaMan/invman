# Benchmark card — `nonstationary_lot_sizing`

**Subfamily:** Dehaybe, Catanzaro & Chevalier 2024 (HenriDeh/DRL_MMULS single-item)

**Difficulty:** `medium` — Single-item scalar order action, but the state must carry a non-stationary forecast window (seasonal / growth / decline over a 32-period horizon), so the policy is forecast-conditioned; there is NO exact global optimum (rolling-DP (s,S) is the strongest baseline, heuristic_to_beat) and verification is against author companion-code CSVs, not a peer-reviewed table.

**Verification tier:** `reference` (re-runs a companion / closed-form / reduced-module number, or a published action)

> Status (manifest, verbatim): verified_rerun (against author COMPANION-CODE TESTBED CSVs, NOT a peer-reviewed EJOR article table; repo flag literature_verified=false is honest)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| dehaybe2024_lostsales_lt2_b5_k10_constant_5 | false | regime:lost_sales, forecast:constant_5, L:2, b:5, K:10, h:1, cv:0.2, H:32, T:104 |
| dehaybe2024_lostsales_lt2_b5_k10_constant_10 (PRIMARY + verification anchor) | false | regime:lost_sales, forecast:constant_10, L:2, b:5, K:10, h:1, cv:0.2, H:32, T:104 |
| dehaybe2024_lostsales_lt2_b5_k10_constant_15 | false | regime:lost_sales, forecast:constant_15, L:2, b:5, K:10 |
| dehaybe2024_lostsales_lt2_b5_k10_seasonal_1 | false | regime:lost_sales, forecast:seasonal_104period, L:2, b:5, K:10 |
| dehaybe2024_lostsales_lt2_b5_k10_seasonal_2 | false | regime:lost_sales, forecast:seasonal_52period, L:2, b:5, K:10 |
| dehaybe2024_lostsales_lt2_b5_k10_seasonal_4 | false | regime:lost_sales, forecast:seasonal_26period, L:2, b:5, K:10 |
| dehaybe2024_lostsales_lt2_b5_k10_growth | false | regime:lost_sales, forecast:linear_growth_5to15, L:2, b:5, K:10 |
| dehaybe2024_lostsales_lt2_b5_k10_decline | false | regime:lost_sales, forecast:linear_decline_15to5, L:2, b:5, K:10 |
| constant_10_rolling_dp_reference (VerificationProblemInstance) | false | verification_anchor, regime:lost_sales, forecast:constant_10 |
| WORKED_EXAMPLE_REFERENCE (Section 4.2, reward -130) | false | regime:backorders, mechanics_self_consistency_only, L:1 |
| retail_like_weekly_trace (practical) | absent (practical dataset, no field) | regime:lost_sales, practical:repo_curated_semi_real, L:2, H:8, T:32, demand:poisson |

## Baselines

**Heuristics**
- simple_s_s (closed-form (s,S): s=quantile at b/(b+h), S=s+EOQ; CV-Normal)
- rolling_dp_s_s (per-period Scarf-style finite-horizon DP, discount 0.99, 32-period tail; Poisson) - STRONGEST
- lead_time_base_stock (repo heuristic, no EOQ batching)

**Exact solver / bound**

none (no exact/global optimum for the rolling-forecast path; rolling_dp_s_s is the strongest baseline, NOT presented as global optimum)

**Published rows**
- author-CSV simple (s,S) constant_10: mean_cost=1832.9142436489014, shortage=0.0029443487165113735 (CV-Normal)
- author-CSV rolling-DP constant_10: mean_cost=1711.741, shortage=0.04793465748308879 (Poisson)
- author-CSV all 8 forecasts carried
- NO PPO/DRL number carried (ppo is a name only; EJOR full text inaccessible)

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | Learned CMA-ES soft tree beats rolling-DP on all 8/8 forecasts by -6.5% to -15.5% (constant_5 1026.3 vs 1214.9 = -15.52%; constant_10 1539.0 vs 1714.1 = -10.22%) |
| `single_seed` | True | no | Learned is cheapest policy on 5/8 instances; on seasonal_2/growth/decline beats DP but trails lead_time_base_stock by +0.35% to +3.46% |
| `multi_seed_mean_std` | False | yes | Heuristic baselines reproduce author-CSV rows within 0.17% (verified ~0.11-0.14%) |

## How to reproduce & compare

**Expected (published) value:** Author companion-code testbed CSV. constant_10: simple (s,S) mean_cost=1832.9142436489014 / shortage 0.0029443487165113735; rolling-DP mean_cost=1711.741 / shortage 0.04793465748308879. Closed-form: simple (s,S) s=33.351246609652, S=47.49338223338295; rolling-DP first-period (28,42).

**Reproduced value (this audit):** constant_10 simple_s_s @25000 reps: 1834.918166 (+0.109%), shortage 0.002871 — within 35.0 cost / 0.01 shortage tol. rolling_dp @25000 reps: 1714.147560 (+0.141%), shortage 0.048469. simple_s_s levels EXACT s=33.351246609652, S=47.493382233383. rolling_dp first-period (28,42) EXACT. growth simple_s_s: 1753.169 (-0.091%). Worked transition -130 confirmed. Non-constant rolling-DP (growth/decline/seasonal) did NOT finish in ~2 min (per-period 32-period DP x104).

**Rerun method / tolerance:** RAYON_NUM_THREADS=4 python via bindings: nonstationary_lot_sizing_simple_s_s_levels; nonstationary_lot_sizing_rolling_dp_s_s_levels; nonstationary_lot_sizing_simulate_policy('simple_s_s',...); nonstationary_lot_sizing_simulate_rolling_dp_policy(...). Constant-forecast cases fast (<5s simple, ~30-60s rolling-DP).

**Reproduce command(s):**

```bash
RAYON_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_literature_benchmark.py --replications 25000
RAYON_NUM_THREADS=4 python -c "import invman_rust as ir; f=[10.0]*136; print(ir.nonstationary_lot_sizing_simple_s_s_levels(f[:32],2,1.0,5.0,10.0,'cv_normal',0.2)); print(ir.nonstationary_lot_sizing_rolling_dp_s_s_levels(f[:32],2,1.0,5.0,10.0,'poisson',0.99,32))"
RAYON_NUM_THREADS=2 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_literature_benchmark.py --learned --tree_depth 2 --leaf_type linear --action_cap 100 --generations 150 --popsize 48 --learned_replications 10000 --output_json /tmp/learned.json
RAYON_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/nonstationary_lot_sizing/run_practical_benchmark.py
cargo test -p invman_rust nonstationary_lot_sizing
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
