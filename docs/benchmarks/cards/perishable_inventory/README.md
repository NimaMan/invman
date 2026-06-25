# Benchmark card — `perishable_inventory`

**Subfamily:** De Moor 2022 Scenario A + Farrington 2025 Table 3

**Difficulty:** `medium` — State is the age-stratified inventory vector (m=2..4 age buckets x leadtime pipeline), so dimensionality grows fast (121 to ~1.3M states); the small m=2/L=1 slice HAS an exact tabular VI true optimum (true_optimum_match_only, De Moor + Farrington Table 3 re-derived), but 28 of 32 Scenario-A rows exceed the 2000-state exact cap and are table-only.

**Verification tier:** `strict` (re-runs a PEER-REVIEWED printed number)

> Status (manifest, verbatim): verified_rerun (genuine re-derivation of 3 independent published quantities for the four m=2/L=1 instances)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| de_moor2022_m2_exp1_l1_cp7_lifo | true (strict_literature_verified) | regime:perishable, issuing:lifo, m:2, leadtime:L1, waste_cost:7, states:121, role:exact_verification, cv:0.5 |
| de_moor2022_m2_exp2_l1_cp7_fifo | true (primary; strict_literature_verified) | regime:perishable, issuing:fifo, m:2, leadtime:L1, waste_cost:7, states:121, role:primary+exact_verification, cv:0.5 |
| de_moor2022_m2_exp4_l1_cp10_fifo | table_only (figure3=None; re-derivable but no verification.rs assertion) | regime:perishable, issuing:fifo, m:2, leadtime:L1, waste_cost:10, states:121, role:autoresearch_extra |
| de_moor2022_m3_exp2_l1_cp7_fifo | table_only | regime:perishable, issuing:fifo, m:3, leadtime:L1, waste_cost:7, states:1331, role:autoresearch_extra |
| de_moor2022_m2_exp6_l2_cp7_fifo | table_only (first instance with genuine in-transit pipeline) | regime:perishable, issuing:fifo, m:2, leadtime:L2, waste_cost:7, states:1331, pipeline:in_transit |
| de_moor2022_m4_exp6_l2_cp7_fifo | table_only (exceeds 2000-state exact cap; Farrington -1432 VI / -1453 base-stock as stored anchors) | regime:perishable, issuing:fifo, m:4, leadtime:L2, waste_cost:7, states:~1.3M, role:practical_benchmark |

## Baselines

**Heuristics**
- base_stock (single level S)
- bsp_low_ew (low-inventory/estimated-waste base-stock with threshold)

**Exact solver / bound**

exact tabular value iteration (value_iteration_mdp.rs; gamma=0.99; midpoint-binned Gamma demand; capped at 2000 states) — re-derives De Moor optimal policy table + best base-stock + Farrington Table 3 VI return.

**Published rows**
- Farrington 2025 Table 3 VI: FIFO -1457+/-59, LIFO -1553+/-61
- De Moor 2022 best base-stock: FIFO S=7, LIFO S=5
- De Moor 2022 9x9 optimal-policy tables FIFO/LIFO
- Farrington 2025 best base-stock FIFO -1474
- 28 of 32 Scenario A rows are TABLE-ONLY (1331..1.77M states), not re-derived
- De Moor DQN/shaped-DQN NOT re-implemented

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `multi_seed_mean_std` | False | yes | Seed-robust exact-anchor FIFO m2/exp2: learned -1458.509+/-0.438 vs base-stock gate -1475.709+/-0.037, +1.166%+/-0.030%, 5/5 seeds beat the gate. |
| `multi_seed_mean_std` | False | yes | Seed-robust exact-anchor LIFO m2/exp1: learned -1553.552+/-0.988 vs base-stock gate -1566.455+/-0.033, +0.824%+/-0.065%, 5/5 seeds beat the gate. |
| `single_seed` | True | no | Learned beats gate on 3 further larger/table-only instances (m3 exp2 +0.70%, m2/L2 exp6 +2.21%, m2 exp4 cp10 +1.44%); these remain single-seed observations. |
| `single_seed` | True | no | exact_slice_report: soft_tree_sigmoid_linear beats best heuristic FIFO ~15.6 units, LIFO ~14 units; soft_tree_linear LIFO worse basin (honest negative) |

## How to reproduce & compare

**Expected (published) value:** Farrington 2025 Table 3 VI: FIFO -1457, LIFO -1553; De Moor S=7 FIFO / S=5 LIFO; 9x9 policy tables; Farrington base-stock FIFO -1474

**Reproduced value (this audit):** FIFO: VI -1457.281 (rounded -1457), S=7, policy table matches=True; LIFO: VI -1552.991 (rounded -1553), S=5, matches=True; FlowNet base-stock FIFO published -1474 / observed -1475 (within tol 1.0)

**Rerun method / tolerance:** python3 -c ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp2_l1_cp7_fifo'); ...('de_moor2022_m2_exp1_l1_cp7_lifo'); ir.perishable_inventory_flownet_policy_verification_summary() — all matches_published_* True THIS audit.

**Reproduce command(s):**

```bash
python3 -c "import invman_rust as ir; s=ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp2_l1_cp7_fifo'); print(s['value_iteration_mean_return_rounded'], s['best_base_stock_level'], s['matches_published_value_iteration_mean_return'], s['matches_published_policy_table'], s['matches_published_base_stock_level'])"
python3 -c "import invman_rust as ir; s=ir.perishable_inventory_exact_mdp_summary('de_moor2022_m2_exp1_l1_cp7_lifo'); print(s['value_iteration_mean_return_rounded'], s['best_base_stock_level'])"
python3 -c "import invman_rust as ir; r=ir.perishable_inventory_flownet_policy_verification_summary(); print(r['summary']['all_observed_targets_within_tolerance'])"
python3 /home/nima/code/ml/invman/scripts/perishable_inventory/autoresearch_perishable_inventory.py --reference de_moor2022_m2_exp2_l1_cp7_fifo --budget smoke --seed 123
python3 /home/nima/code/ml/invman/scripts/perishable_inventory/run_exact_slice_benchmark.py
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
