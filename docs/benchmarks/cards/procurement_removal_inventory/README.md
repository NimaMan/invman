# Benchmark card — `procurement_removal_inventory`

**Subfamily:** control-only reduction of Maggiar & Sadighian 2017

**Difficulty:** `easy` — Low-dim single-item state (inventory + returnable buffer) with a small (order, remove) action; the reduced verifier has an EXACT backward-induction DP true optimum (31.78026 to 1e-10, true_optimum_match_only) and the structural-optimum interval-stock heuristic is the comparator — well-posed despite no published number.

**Verification tier:** `faithful` (faithful_unverified (validated only vs the repo's own exact DP))

> Status (manifest, verbatim): faithful_unverified (env faithful-to-STRUCTURE; what re-ran is repo-native self-consistency; NO published number exists)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| maggiar2017_style_fixed_returnability (PRIMARY) | false (repo_native_instance_not_verified_against_literature) | regime:lost_sales, demand:poisson, mean:4.0, periods:16, init_inv:5, returnable_cap:2, removal_inactive, discount:0.99 |
| removal_active_returnability | false (repo_native_removal_active_instance) | regime:lost_sales, demand:poisson, mean:3.0, periods:16, init_inv:12, init_returnable:8, holding:1.0, removal_channel_binds, discount:0.99 |
| VERIFICATION_PROBLEM_INSTANCE (reduced exact-DP) | false (repo_exact_solver_not_verified_against_literature) | regime:lost_sales, demand:discrete_support[0,1,2,3], periods:5, init_inv:2, returnable_cap:1, discount:0.99, exact_DP_solvable |

## Baselines

**Heuristics**
- interval_stock (order_up_to, remove_down_to) — structural optimum form per Maggiar & Sadighian Theorem 3.4; grid-tuned
- returnability_buffer_interval_stock (order_up_to, remove_down_to, returnable_buffer)

**Exact solver / bound**

finite_horizon_dp.rs solve_optimal_policy — exact bounded backward-induction DP, ONLY on the small discrete-support VERIFICATION_PROBLEM_INSTANCE (periods=5); repo-native optimum, NOT published. The two benchmark instances (periods=16 Poisson) are NOT solved exactly.

**Published rows**
- NONE. Maggiar & Sadighian 2017 only numerical example (Sec 7 Table 1) is a pricing-coupled NPV surface ~84000, not a control-only cost. Maggiar et al. 2025 NeurIPS reports returns family qualitatively only (Fig 23).

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | primary: soft_tree (depth-2) = 358.218 essentially TIES interval-stock 358.107 (0.03% behind). NOT a beat. |
| `single_seed` | True | no | removal_active: soft_tree = 251.727 is 3.1% BEHIND interval-stock 244.117. NOT a beat — heuristic wins. |
| `none` | False | no | exact DP (reduced verifier) optimal 31.7802611137 dominates both heuristics (interval_stock 34.164, returnability_buffer 38.766) — re-run, holds. |

## How to reproduce & compare

**Expected (published) value:** none — neither cited paper exposes a public procurement-removal control-only cost row

**Reproduced value (this audit):** exact-DP optimal_discounted_cost = 31.78026111369698 (README claim 31.7802611137, match to 1e-10). Heuristic rows reproduced exactly: primary interval_stock (6,6) = 358.1067286254911; removal_active interval_stock (4,9) = 244.11666203081566. Worked-transition: period_cost=10.5 (assertions pass).

**Rerun method / tolerance:** invman_rust.procurement_removal_inventory_exact_dp_summary(); procurement_removal_inventory_simulate_policy('interval_stock', [6,6]/[4,9], seeds=range(500000,504096), discount=0.99); procurement_removal_inventory_step(...) worked transition.

**Reproduce command(s):**

```bash
python -c "import invman_rust as r; s=r.procurement_removal_inventory_exact_dp_summary(); print(s['optimal_discounted_cost'], s['optimal_first_action'])"
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py --train --eval_seeds 4096 --generations 80 --output_json outputs/procurement_removal_inventory/benchmark.json
python -c "import invman_rust as r; print(r.procurement_removal_inventory_step(4,2,3,2,4,2,6.0,4.0,1.0,0.5,9.0))"
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
