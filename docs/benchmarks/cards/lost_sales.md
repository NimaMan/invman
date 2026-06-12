# Benchmark card — `lost_sales`

**Subfamily:** vanilla (Zipkin 2008) + fixed_order_cost (Bijvank 2015)

**Difficulty:** `easy` — Low-dim scalar state (inventory position + short pipeline), single scalar order action; fixed-cost subfamily has an in-repo exact average-cost VI optimum (true_optimum_match_only) and vanilla carries published Zipkin DP optima with myopic/base-stock heuristics to beat — a well-posed, solver-anchored benchmark.

**Verification tier:** `strict` (re-runs a PEER-REVIEWED printed number)

> Status (manifest, verbatim): verified_rerun

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| vanilla_l4_p4_poisson5 (alias lit_poisson_p4_l4) | true | regime:lost_sales, subproblem:vanilla, demand:poisson, mu:5, leadtime:L4, h:1, p:4, anchor:Zipkin2008_Table3a |
| lit_poisson_p19_l4 | absent (reference_costs.rs has no per-row flag; source=literature, carries published optimal 8.89) | regime:lost_sales, subproblem:vanilla, demand:poisson, leadtime:L4, p:19_high_penalty |
| lit_poisson_p4_l6 / l8 / l10 | absent (source=literature or literature+computed but heuristic cells are full-precision repo-computed, e.g. myopic1=5.4140775) | regime:lost_sales, subproblem:vanilla, demand:poisson, leadtime:L6/L8/L10_deep_pipeline |
| lit_geometric_p4_l4 .. lit_geometric_p19_l10 (8 rows) | absent (source=literature+computed; M1/SVBS repo-computed) | regime:lost_sales, subproblem:vanilla, demand:geometric, cv:high, leadtime:L4-L10, p:4_and_19 |
| lit_mmpp2_pos_* / lit_mmpp2_neg_* (16 rows) | absent (source=computed; order qty on stationary marginal, cost on true MMPP2; no capped_base_stock) | regime:lost_sales, subproblem:vanilla, demand:markov_modulated_poisson2, autocorr:positive_p00=p11=0.9 / negative_p00=p11=0.1, leadtime:L4-L10, p:4_and_19 |
| bijvank2015_table1_l2_p14_k5 | true | regime:lost_sales, subproblem:fixed_order_cost, demand:poisson, mu:5, leadtime:L2, h:1, p:14, K:5, anchor:Bijvank2015_Table1, exact_solvable |
| fixed_order_cost full experiment grid 'lost_sales_style_full_grid_mu5' | absent (README: larger grids not yet literature-verified) | regime:lost_sales, subproblem:fixed_order_cost, demand:{poisson,geometric,mmpp2_pos,mmpp2_neg}, leadtime:L2/L4/L6/L8/L10, p:4_and_19, K:5_and_25, size:80_instances |

## Baselines

**Heuristics**
- vanilla: Myopic-1 (Zipkin 'Myopic')
- vanilla: Myopic-2
- vanilla: Standard Vector Base Stock (SVBS, Morton 1969/71)
- vanilla: better/capped base-stock (Zipkin 'Better vector base-stock' 4.80, corroborated Xin 2021)
- fixed-cost: (s,S)
- fixed-cost: (s,nQ)
- fixed-cost: modified (s,S,q)

**Exact solver / bound**

fixed_order_cost: exact_value_iteration.rs — average-cost bounded DP / relative value iteration over the lost-sales pipeline (exact within inventory-position cap). VANILLA: NO in-repo exact solver — optima 4.73/8.89/5.31/etc. are carried published Zipkin DP values, NOT recomputed.

**Published rows**
- VANILLA Zipkin 2008 Table 3(a) L=4 Poisson(5) h=1 p=4: Optimal 4.73, Myopic 5.06, Myopic-2 4.82, SVBS 5.83, Better-VBS 4.80
- FIXED-COST Bijvank 2015 Table 1 R=1 L=2 h=1 p=14 K=5 Poisson(5): Optimal 11.46, (s,S)=(17,23) 11.62, (s,nQ)=(17,7) 11.56, modified (s,S,q)=(17,23,7) 11.50
- No published DRL row in the lost-sales tables (Gijsbrechts 2022 carried as context only)

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | Vanilla surface: learned policies best in 22/24 reported instances; on L4-Poisson4 depth-2 linear-leaf soft tree 4.7537 vs myopic2 4.8186 (-1.20%) |
| `single_seed` | True | no | Fixed-cost surface (48 instances): learned competitive/winning vs (s,S)/(s,nQ)/(s,S,q); canonical L4 Poisson K5 p4 learned ~8.73 vs heuristic ~9.20 |
| `multi_seed_mean_std` | False | yes | Canonical vanilla L4-Poisson p4 depth-2 soft tree within eval noise of published optimum 4.73 (learned 4.7537) |

## How to reproduce & compare

**Expected (published) value:** vanilla L4 Poisson5: optimal 4.73, myopic1 5.06, myopic2 4.82, svbs 5.83; fixed-cost Bijvank Table 1: optimal 11.46, (s,S) 11.62, (s,nQ) 11.56, modified (s,S,q) 11.50

**Reproduced value (this audit):** VANILLA (horizon=100k, seed=123): myopic1=5.0569, myopic2=4.8208, svbs=5.8153 (within ~0.015). FIXED-COST (cap=24, exact DP): optimal=11.4631 (gap +0.0031, first_action=8), (s,S)=11.6181, (s,nQ)=11.5552, modified=11.4974 (gaps <0.005). Fixed-cost is a genuine EXACT solver match; vanilla optimum 4.73 is a carried Zipkin value (only the 3 heuristic rows re-run). CAVEAT: only vanilla_l4_p4_poisson5 carries true Zipkin numbers; rest of 33-instance grid is repo-computed.

**Rerun method / tolerance:** python -c via invman_rust: (1) ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995); (2) ir.lost_sales_fixed_order_cost_exact_literature_summary('bijvank2015_table1_l2_p14_k5',24). Both <1 min.

**Reproduce command(s):**

```bash
python -c "import invman_rust as ir; print(ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995))"
python -c "import invman_rust as ir; import json; print(json.dumps(ir.lost_sales_fixed_order_cost_exact_literature_summary('bijvank2015_table1_l2_p14_k5',24), default=str))"
python -c "import invman_rust as ir; [print(n, ir.lost_sales_reference_costs(n)['source'], ir.lost_sales_reference_costs(n)['costs']) for n in ir.lost_sales_reference_instance_names()]"
python -c "import invman_rust as ir; g=ir.lost_sales_fixed_order_cost_expand_experiment_grid('lost_sales_style_full_grid_mu5'); print(len(g), g[0]['name'], g[-1]['name'])"
python /home/nima/code/ml/invman/scripts/lost_sales/validate_reference_instance.py --num_seeds 3
python /home/nima/code/ml/invman/scripts/lost_sales/benchmark_full_suite.py --seed 42 --eval_seeds 10
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._

