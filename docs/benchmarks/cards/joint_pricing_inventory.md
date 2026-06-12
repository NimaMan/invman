# Benchmark card — `joint_pricing_inventory`

**Subfamily:** finite-horizon joint pricing-and-ordering (price-setting newsvendor)

**Difficulty:** `easy` — Low-dim (period, inventory) state with a small joint (price-index, order-quantity) action over a short horizon; the T=5 verifier has an EXACT backward-induction DP true optimum (true_optimum_match_only, optimal -33.178 reproduced incl. an independent Python DP), well-posed despite having no published per-instance number.

**Verification tier:** `faithful` (faithful_unverified (validated only vs the repo's own exact DP))

> Status (manifest, verbatim): faithful_unverified (env faithful, both repo-internal anchors re-ran and matched, but neither is a PUBLISHED per-instance number)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| VERIFICATION_PROBLEM_INSTANCE (Qin-2022-labeled exact verifier) | false | regime:lost_sales, horizon:T5, price_ladder:3_levels[7,9,11], discount:0.99, max_order_quantity:4, exact_DP_feasible:true, leadtime:L0, salvage:1.0 |
| PRIMARY_REFERENCE_INSTANCE / zhou2022_style_price_ladder | false | regime:lost_sales, horizon:T18, price_ladder:3_levels[8,10,12], demand:poisson_price_dependent[4.0,2.6,1.6], discount:0.99, max_order_quantity:6, no_exact_optimum:true, leadtime:L0, salvage:1.0 |

## Baselines

**Heuristics**
- static_price_base_stock (order-up-to + fixed price index)
- inventory_sensitive_base_stock (order-up-to + markdown threshold)

**Exact solver / bound**

finite_horizon_dp.rs solve_optimal_policy — backward-induction exact DP over (period, inventory), feasible on T=5 verifier; exposed via joint_pricing_inventory_exact_dp_summary().

**Published rows**
- NONE reproduced. references.rs carries ZHOU_2022 / QIN_2022 / PRICE_SETTING_NEWSVENDOR_ANCHOR labels but ALL benchmark_policies entries are labels-only; none implemented, no numeric published row stored. Both instances literature_verified=false.

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `none` | False | no | Exact-DP-anchored profit gaps on verifier: static_price_base_stock 2.02%, inventory_sensitive 16.83%. Re-ran exactly. |
| `single_seed` | True | no | Trained depth-2 oblique/linear soft tree beats best heuristic by +25.15% profit on primary 18-period Poisson instance (216.060 vs 171.513), 4096 eval seeds. |

## How to reproduce & compare

**Expected (published) value:** No published per-instance optimal-profit number. Verification rests on (a) analytical critical-fractile newsvendor, (b) repo exact DP optimal -33.178121049724 first action (2,1) (self-consistency).

**Reproduced value (this audit):** exact DP optimal -33.178121049724, first action (2,1); static -32.50820139235; inventory_sensitive -27.594377111812527. Critical-fractile y*=(3,2,2) matched by env brute force. Independent hand-coded Python DP reproduced -33.178121049724 within 1e-9. Learned benchmark: soft tree -216.0595, static -172.6349, inv-sensitive -171.5135, +25.15% profit.

**Rerun method / tolerance:** invman_rust.joint_pricing_inventory_exact_dp_summary(); critical-fractile via joint_pricing_inventory_step + joint_pricing_inventory_exact_verification_instance; independent lru_cache Python DP; python scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py --replications 4096 --seed 777000.

**Reproduce command(s):**

```bash
python -c "import invman_rust; print(dict(invman_rust.joint_pricing_inventory_exact_dp_summary()))"
python -c "import invman_rust; ref=dict(invman_rust.joint_pricing_inventory_exact_verification_instance()); pl=list(ref['price_levels']); [print(pi, invman_rust.joint_pricing_inventory_step(0,q,pi,d,pl,4.0,0.5,5.0)) for pi in range(3) for q in range(5) for d in range(4)]"
python scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py --replications 4096 --seed 777000
python scripts/joint_pricing_inventory/validate_against_exact_dp.py --simulation_replications 512 --simulation_seed 123
python scripts/joint_pricing_inventory/train_soft_tree_reference.py --depth 2 --leaf_type linear --seed 123 --eval_seeds 2048
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._

