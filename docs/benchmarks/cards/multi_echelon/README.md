# Benchmark card — `multi_echelon`

**Subfamily:** serial (Clark-Scarf), assembly (Rosling), divergent_special_delivery (Van Roy/Gijs), production_assembly_distribution_network (Pirhooshyaran-Snyder), general_backorder_fixed_cost (Geevers/Kunnumkal-Topaloglu)

**Difficulty:** `hard` — Umbrella entry; difficulty is per-subfamily — serial=medium (low-dim chain WITH an exact Clark-Scarf true optimum, but multi-stage coordination), everything else=hard (high-dim networks, allocation/topology, mostly self-consistent or bound_gap comparators, several rows not reproduced). Whole entry rated HARD to its hardest members.

**Difficulty by subfamily:**
- `serial` → `medium` — Low-dim Clark-Scarf chain; EXACT recursive-newsvendor true optimum (true_optimum_match_only) mirrors stockpyl — easiest of the family, but multi-stage echelon coordination keeps it above the single-node easy problems.
- `assembly` → `hard` — Multiple components must synchronize; only verified-by-equivalence via Rosling reduction (literature_verified=false), no direct published anchor.
- `divergent_special_delivery` → `hard` — Warehouse-to-many-retailer allocation + special-delivery mode; comparator is best constant base-stock by grid (heuristic_to_beat) and published A3C savings are cross-protocol context (debt D1).
- `production_assembly_distribution_network` → `hard` — Arbitrary production/assembly/distribution topology; only single-node analytical rows reproduced, general-network gate is the env's OWN base-stock (self-consistent), no published cost reproduced.
- `general_backorder_fixed_cost` → `hard` — 4-supplier/4-warehouse/5-retailer networks, high state/action dimensionality; set1/KT reproduced but set2/set3 are +223% NOT reproduced (debt D2); heuristic_to_beat with cross-protocol PPO context.

**Verification tier:** `mixed` (umbrella — tiers differ per sub-family (see map))

**Verification tier by subfamily:**
- `serial` → `strict`
- `assembly` → `faithful`
- `divergent_special_delivery` → `strict`
- `production_assembly_distribution_network` → `faithful`
- `general_backorder_fixed_cost` → `strict`

**Tier note:** Per-subfamily tiers (ledger-reconciled): serial=strict (Clark-Scarf Snyder-Shen Ex6.1 47.65 re-run); divergent_special_delivery=strict (Van Roy constant-base-stock 51.7/1302/1449 re-run within 2%; A3C savings rows = snapshot debt D1, not the tier basis); general_backorder_fixed_cost=strict for set1+KT (Geevers set1 10467, Kunnumkal-Topaloglu 4059 re-run) but set2/set3 (+223%) are an UNREPRODUCED debt (D2); assembly=faithful (verified-by-equivalence via Rosling reduction only, no direct anchor); production_assembly_distribution_network=faithful (only single-node analytical rows reproduce; general-network gate is the env's own base-stock).

> Status (manifest, verbatim): verified_rerun (serial + gbk set1/KT + divergent const-base-stock + padn single-node); faithful_unverified (assembly, padn general-network); snapshot_only_not_rerun (divergent A3C relative rows, gbk set2/3, van_oers padn table-only)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| serial:snyder_shen_example_6_1 | true (the one genuinely published anchor) | subfamily:serial, regime:backorder, N:3, demand:normal(5,1), leadtime:L[2,1,1], echelon_holding:[2,2,3], penalty:37.12, published_optimum:47.65 |
| serial:poisson_N1_N2_N3 | reference-implementation-verified (matches stockpyl, NOT paper-printed) | subfamily:serial, regime:backorder, N:1/2/3, demand:poisson(5), leadtime:L0=1, reference_impl:stockpyl.ssm_serial, optima:4.2208/16.7978/72.0435 |
| serial:two_stage_normal_and_five_stage_normal_poisson | reference-implementation-verified (stockpyl-derived) | subfamily:serial, N:2/5, demand:normal/poisson, optima:166.2705/225.8672/226.8458, stockpyl:problem_6_1/6_2a/6_2b |
| assembly:two/three_component + heterogeneous (3 instances) | false (guarded by no_assembly_instance_is_literature_verified) | subfamily:assembly, regime:backorder, components:2/3/2, demand:poisson(5)/poisson(4), solver_derived_cost:22.759/52.536/27.530 |
| divergent:van_roy1997_simple_problem | false | subfamily:divergent_special_delivery, regime:hybrid_lost_sales_special_delivery, K:1, mode:van_roy_1997, published_const_base_stock:51.7, published_ndp:52.6 |
| divergent:van_roy1997_case_study1 (Gijs setting1) | false | subfamily:divergent_special_delivery, K:10, leadtime:lw2_lr2, published_const_base_stock:1302, published_ndp:1179, published_a3c_savings:8.95% |
| divergent:van_roy1997_case_study2 (Gijs setting2) | false | subfamily:divergent_special_delivery, K:10, leadtime:lw5_lr3, published_const_base_stock:1449, published_ndp:1318, published_a3c_savings:12.09% |
| divergent:gijsbrechts2022_setting1 / setting2 (paper-faithful search targets) | false | subfamily:divergent_special_delivery, K:10, mode:gijs_2022_pre_shipment_eq2, demand_mean:5/0, no_published_absolute_row, primary_reference_instance:setting2 |
| padn:pirhooshyaran2021_single_node_cases_1-7 | false flag on reference instances, BUT single-node analytical rows reproduced by re-run | subfamily:production_assembly_distribution_network, node_mode:single, regime:newsvendor, published_analytical_OUL:10.67..106.74, published_cost:12.71/25.42/63.56/127.11 |
| padn:serial_case3 + mixed_scn_fig1_table5 + pure_assembly_network | false (faithful-but-no-published-anchor) | subfamily:production_assembly_distribution_network, gates:60.24(serial)/297.69(mixed)/283.34(pure_assembly), table_only_serial_optima |
| gbk:geevers2023_general_set1 (CardBoard) | true | subfamily:general_backorder_fixed_cost, regime:backorder, topology:4supplier_4warehouse_5retailer, demand:poisson(15), published_benchmark:10467, published_ppo_best:8714 |
| gbk:geevers2023_general_set2 / set3 (order-per-edge) | false (both set2 and set3) | subfamily:general_backorder_fixed_cost, topology:4w_5r, action:order_per_edge, published_benchmark:4797, published_ppo_best:4175/3935, NOT_reproduced(+223%) |
| gbk:kunnumkal_topaloglu_divergent | true | subfamily:general_backorder_fixed_cost, regime:backorder, topology:1supplier_1warehouse_3retailer, published_benchmark:4059, published_drl:3724_cross_protocol |

## Baselines

**Heuristics**
- serial: exact Clark-Scarf echelon base-stock (optimal); newsvendor-per-echelon; lead-time-mean
- assembly: Rosling-reduced serial echelon base-stock
- divergent: best constant base-stock by grid search; min_shortage allocation
- padn: best pairwise base-stock gate (env's own, NOT a published optimum); single-node newsvendor
- gbk: constant node-base-stock at published levels

**Exact solver / bound**

serial: exact Clark-Scarf recursive-newsvendor decomposition (TRUE optimum, mirrors stockpyl). divergent: bounded finite-horizon DP (repo-internal verifier, not published). padn: tiny finite_horizon_dp on a small serial network (repo-native). assembly: reuses serial exact solver via Rosling reduction. gbk: none (heuristic-only).

**Published rows**
- serial: Snyder & Shen Ex6.1 optimal 47.65
- divergent: Van Roy const base-stock 51.7/1302/1449; best NDP 52.6/1179/1318; Gijs A3C savings 8.95%+/-0.13% / 12.09%+/-0.39%
- padn: Pirhooshyaran single-node OUL/cost (7 cases); serial Clark-Scarf optima (catalog-only)
- gbk: Geevers set1 10467 / PPO best 8714; set2 4797; set3 4797; Kunnumkal-Topaloglu 4059 / DRL 3724

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | False | no | serial: warm-started echelon soft tree ties Clark-Scarf optimum 47.6554 vs 47.65 (+0.011%). MATCH-only. |
| `multi_seed_mean_std` | False | yes | divergent setting1: direct-level learned seed mean 776.15+/-14.27 vs best constant base-stock gate 910.34+/-0.51 -> 14.74%+/-1.60% cost reduction; 5/5 seeds beat the gate. A3C 8.95% remains cross-protocol context. |
| `multi_seed_mean_std` | False | yes | divergent setting2: direct-level learned seed mean 1001.07+/-25.63 vs best constant base-stock gate 1138.04+/-0.43 -> 12.04%+/-2.26% cost reduction; 5/5 seeds beat the gate. A3C 12.09% remains cross-protocol context. |
| `multi_seed_mean_std` | False | yes | padn serial case3: five-seed audit is parity (mean 58.90 vs env gate 60.24, 4/5 below); single-run wins are illustrative only. |
| `multi_seed_mean_std` | False | yes | padn pure-assembly: five-seed audit is parity/loss (289.16 ± 14.20 vs env gate 283.34, 2/5 below); single-run win is illustrative only. |
| `multi_seed_mean_std` | False | yes | padn mixed distribution-assembly: residual base-stock-backbone head beats the env-own gate 297.69 with 291.136+/-2.78 over 5 seeds (-2.20%, 5/5 below gate). The older vector/flow-head audit remains parity at 306.10+/-22.89 (+2.82%, 4/8 below). Not a published-number beat. |
| `multi_seed_mean_std` | False | yes | gbk set1 (CardBoard): learned seed mean 7772.10+/-142.21 vs reproduced gate 10354.82 -> 24.94%+/-1.37% cost reduction; 5/5 seeds beat the gate. |
| `multi_seed_runs_no_aggregate_json` | False | no | gbk Kunnumkal-Topaloglu: five full-budget seed rows all beat reproduced gate 3930.4 by about 36.7% (learned 2469.1..2498.4); published DRL 3724 remains cross-protocol context. |

## How to reproduce & compare

**Expected (published) value:** serial Snyder-Shen Ex6.1 = 47.65; divergent Van Roy const base-stock 51.7/1302/1449; padn single-node 12.71..127.11 (7 cases); gbk set1 = 10467, KT = 4059; gbk set2/3 = 4797 (NOT reproduced)

**Reproduced value (this audit):** serial: 47.6654 (Ex6.1), Poisson N1/N2/N3 = 4.2211/16.7983/72.0467, 5-stage 225.867/226.846, 2-stage 166.271 (ALL re-run). divergent: Van Roy all 3 rows within 2% (51.77/1284.70/1437.96); A3C relative rows NOT reproduced. padn: single-node 7 cases to ~0.005 abs. gbk: set1 = 10384.9 (-0.78%), KT = 3933.3 (-3.1%), set2 = 15497 (+223%, NOT reproduced). assembly: independent Rosling structural verifier passes, but no learned-policy env binding.

**Rerun method / tolerance:** ir.multi_echelon_serial_exact_normal_solution([3,2,2],[1,1,2],37.12,5,1)->47.6654; ir.multi_echelon_serial_exact_poisson_solution(...)->72.0467; ir.multi_echelon_van_roy_reproduction_summary(repo_audit_replications=20,seed=1); ir.production_assembly_distribution_network_literature_benchmark_summary(serial_replications=10000,seed=1234); ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('geevers2023_general_set1',replications=200,seed=1234).

**Reproduce command(s):**

```bash
python -c "import invman_rust as ir; print(ir.multi_echelon_serial_exact_normal_solution([3,2,2],[1,1,2],37.12,5.0,1.0))"
python -c "import invman_rust as ir; print(ir.multi_echelon_serial_exact_poisson_solution([3,2,2],[1,1,2],37.12,5.0))"
python -c "import invman_rust as ir,json; print(json.dumps(ir.multi_echelon_van_roy_reproduction_summary(repo_audit_replications=20,seed=1),default=str))"
python -c "import invman_rust as ir,json; print(json.dumps(ir.multi_echelon_gijs_relative_verification_summary(repo_audit_replications=20,seed=1),default=str))"
python -c "import invman_rust as ir,json; print(json.dumps(ir.production_assembly_distribution_network_literature_benchmark_summary(serial_replications=10000,seed=1234),default=str))"
python -c "import invman_rust as ir; print(ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('geevers2023_general_set1',replications=200,seed=1234)['mean_cost'])"
python -c "import invman_rust as ir; print(ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('kunnumkal_topaloglu_divergent',replications=500,seed=1234)['mean_cost'])"
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/multi_echelon/seed_robust_divergent_multi_echelon.py --reference gijsbrechts2022_setting1 --budget full --designs direct_level --depths 2 3 --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/multi_echelon/seed_robust_divergent_multi_echelon.py --reference gijsbrechts2022_setting2 --budget full --designs direct_level --depths 2 3 --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2
python scripts/multi_echelon_serial/benchmark_serial_clark_scarf.py
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/production_assembly_distribution_network/autoresearch_mixed_distribution_assembly_network.py --budget full --warm_start_flow 10 --seed 7 --run_tag mixed_flow10_verify
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python policy_search/agentic/evaluate_policy_spec_padn.py --spec policy_search/agentic/specs/padn_explore_best.json --problem production_assembly_distribution_network --instance 0 --seeds 5 --budget full
python scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py --reference kunnumkal_topaloglu_divergent --budget full
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
