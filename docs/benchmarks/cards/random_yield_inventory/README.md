# Benchmark card — `random_yield_inventory`

**Subfamily:** all-or-nothing batch yield (structural match to Yan et al. 2026)

**Difficulty:** `easy` — Low-dim single-item state (inventory + short pipeline) with a scalar order action under all-or-nothing yield; the reduced L=2 slice has an EXACT discounted finite-horizon DP true optimum (40.0599, true_optimum_match_only) enumerating the demand x yield tree, with linear-inflation / weighted-newsvendor heuristics as comparators.

**Verification tier:** `faithful` (faithful_unverified (validated only vs the repo's own exact DP))

> Status (manifest, verbatim): faithful_unverified (self-consistent-only; reproduced number is repo's OWN exact-DP anchor, NOT a literature number)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| VERIFICATION_PROBLEM_INSTANCE (exact-DP slice) | false | regime:backlog, yield:all_or_nothing, p:0.75, leadtime:L2, horizon:finite_T5, demand:discrete_6pt, gamma:0.99, h:1, b:9, max_order_cap:8, init_inv:4, init_pipeline:[3,2] |
| PRIMARY_REFERENCE_INSTANCE = yan2026_style_lt2_p075_discounted | false | regime:backlog, yield:all_or_nothing, p:0.75, leadtime:L2, horizon:finite_T12, demand:poisson_mean4, gamma:0.99, h:1, b:9, c:1, init_inv:6, init_pipeline:[4,3] |
| YAN_2026_SMALL_SCALE_FAMILY (catalog-only) | absent (do_not_use_for_repo_assertions) | yield:all_or_nothing, horizon:finite_discounted, model_match:exact, access:preview_only, numbers_available:false |
| CHEN_2018_WNH_FAMILY (catalog-only) | absent (do_not_use_for_repo_assertions) | yield:all_or_nothing, policy:weighted_newsvendor, access:bibliographic_only, numbers_available:false |
| INDERFURTH_2015 binomial/proportional grids (different yield model) | absent (related_model_aggregate_only) | yield:binomial_or_proportional, horizon:infinite_avg_cost, model_match:partial/special_case, numbers_available:true_but_different_model |

## Baselines

**Heuristics**
- linear_inflation (LIR): q=(1/p)*(S-X)^+; on VERIFICATION cost 47.7138 (+19.1% vs optimum); on PRIMARY 203.619 +/- 123.769
- weighted_newsvendor (WNH): yield-weighted gap, NOT inflated by 1/p (open fidelity question); VERIFICATION 60.3936 (+50.8%); PRIMARY 222.436 +/- 66.918

**Exact solver / bound**

Exact reduced finite-horizon discounted DP (finite_horizon_dp.rs::solve_optimal_policy), full enumeration over demand x yield tree; lead_time==2 only, discrete demand, capped action. Optimal on VERIFICATION = 40.0599, first action 4.

**Published rows**
- NONE usable: Yan 2026 / Chen 2018 paywalled, no public per-instance table. Inderfurth & Kiesmueller 2015 publishes numbers but for a DIFFERENT yield model (carries over only the F=1/p inflation factor). All three citations metadata-verified 2026-05.

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | README/experiments: learned soft-tree (depth 3, 600ep) = 196.661, beats LIR (203.619) by 3.4% and WNH (222.436) by 11.6% on PRIMARY, 2000 seeds |
| `multi_seed_mean_std` | True | no | Seed-robust outputs: d1_linear b8 (800ep) over 4 seeds {123,456,789,2026} = 196.73 +/- 3.35; ALL 4 beat LIR (mean +4.25% +/- 1.67%) |
| `single_seed` | True | no | Single-config saved runs contradict headline: tree_primary_d3_linear.json soft-tree=307.01 (WORSE than LIR by -48.7%); only the d1 b8 800-ep config beats LIR |

## How to reproduce & compare

**Expected (published) value:** none (no public per-instance number; Yan 2026 / Chen 2018 paywalled; Inderfurth 2015 different model)

**Reproduced value (this audit):** exact-DP slice: optimal_discounted_cost=40.05989760985441, first action 4, LIR=47.71379457283354, WNH=60.39357514301890 — matches README repo-native anchor to full precision. Simulation slice: LIR=203.619+/-123.769, WNH=222.436+/-66.918 over 2000 seeds — matches README exactly.

**Rerun method / tolerance:** invman_rust.random_yield_inventory_exact_dp_summary() (0.15s); invman_rust.random_yield_inventory_policy_discounted_cost_summary(policy_name in {linear_inflation,weighted_newsvendor}, ref=random_yield_inventory_primary_reference_instance(), seeds=range(123,2123), demand_distribution='poisson').

**Reproduce command(s):**

```bash
python -c "import invman_rust, json; print(json.dumps({k:v for k,v in invman_rust.random_yield_inventory_exact_dp_summary().items() if k!='verification_reference'}, indent=1))"
python /home/nima/code/ml/invman/scripts/random_yield_inventory/validate_against_exact_dp.py --simulation_seeds 256
python /home/nima/code/ml/invman/scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py
python /home/nima/code/ml/invman/scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py --train_soft_tree --depth 1 --leaf_type linear --training_episodes 800 --es_population 16 --train_seed_batch 8 --seed 456
python -c "import json,statistics; fs=['/home/nima/code/ml/invman/outputs/random_yield_inventory/tree_primary_d1_linear_b8_s%d.json'%s for s in (123,456,789,2026)]; st=[json.load(open(f))['evaluation']['soft_tree']['mean_cost'] for f in fs]; print('soft_tree %.2f +/- %.2f'%(statistics.mean(st),statistics.pstdev(st)))"
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
