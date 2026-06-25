# Benchmark card — `vendor_managed_inventory`

**Subfamily:** Sui, Gosavi & Lin 2010 consignment VMI (reduced single-retailer slice + truck-dispatch family)

**Difficulty:** `medium` — Two-tier DC+retailer consignment state with a coupled (replenish-DC, ship-to-retailer) action and a capacity-limited shipment, raising it above single-node problems; an exact reduced DP exists in Rust but has NO Python binding (optimality ceiling unexposed), so reported gaps are learned-vs-heuristic and the only open published numbers are an instructor handout (not literature).

**Verification tier:** `faithful` (faithful_unverified (validated only vs the repo's own exact DP))

> Status (manifest, verbatim): faithful_unverified (NO peer-reviewed paper number reproduced; handout + env step + heuristic baseline re-run confirmed, but a handout is NOT literature verification)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| PRIMARY_REFERENCE_INSTANCE / giannoccaro2010_style_single_retailer | false (SUI_GOSAVI_LIN_2010_REFERENCE.literature_verified=false) | regime:lost_sales, single_retailer, periods:24, demand:poisson_mean2.5, stockout:5.0, dc_capacity:10, max_shipment:5, discount:0.99, repo_chosen_no_published_anchor |
| low_penalty | false | regime:lost_sales, stockout:2.0, perturbation_of_primary |
| high_penalty | false | regime:lost_sales, stockout:9.0, perturbation_of_primary, widest_learned_loss |
| low_demand | false | regime:lost_sales, demand:poisson_mean1.5, perturbation_of_primary |
| high_demand | false | regime:lost_sales, demand:poisson_mean3.5, perturbation_of_primary |
| VERIFICATION_PROBLEM_INSTANCE (exact-DP verifier) | false (repo-native self-consistency verifier) | regime:lost_sales, periods:5, discrete_demand_support:{0,1,2,3}, discount:0.99, small_enough_for_exact_DP |
| SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE (newsvendor handout) | false (instructor TEACHING HANDOUT, NOT peer-reviewed) | compound_poisson, single_retailer_single_product, cycle_time:{30,40,50}, newsvendor_order_up_to |
| SUI_GOSAVI_LIN_2010_CASE_DEFINITIONS (8-case truck-dispatch) | false (repo-constructed interpretation; does not reproduce paywalled table) | truck_dispatch, multi_retailer:10, multi_product:2, continuous_time, structural_interpretation_not_transcribed |

## Baselines

**Heuristics**
- retailer_base_stock (grid-tuned level)
- dc_reserve_base_stock (grid-tuned level x DC reserve)
- paper_mean_demand (MDH order-up-to, truck-dispatch)
- paper_newsvendor (newsvendor order-up-to + truck allocation)

**Exact solver / bound**

finite_horizon_dp::solve_optimal_policy on reduced discrete-demand single-retailer instance — EXISTS in Rust, used only by in-crate dominance test; NOT exposed as a Python binding, so NOT used as the benchmark optimality ceiling.

**Published rows**
- NONE for the reduced slice (repo-chosen instance, no published number)
- Gosavi instructor HANDOUT newsvendor: MDH S=15, six-sigma S=31.53, newsvendor S=26.96 (reproduced exactly, but a handout)
- Peer-reviewed Sui/Gosavi/Lin 2010 RL-vs-newsvendor profit table: PAYWALLED, not carried

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | low_penalty FLIPS to a clean WIN: learned soft tree (d3/t0.1/warm-start) = 102.69 vs tuned heuristic 103.01, gap -0.31%. |
| `single_seed` | True | no | primary loss closed from -1.76% to statistical tie (+0.05%); high_penalty -2.40% -> +0.30%; high_demand -0.91% -> +1.12% (single config). |
| `single_seed` | False | no | README baseline: learned (constant leaf, no warm-start, d2) LOSES on 4/5 (primary -1.76%, low_penalty -0.16%, high_penalty -2.40%, high_demand -0.91%); marginally wins low_demand (+0.10%). |

## How to reproduce & compare

**Expected (published) value:** Paper results table paywalled / no numeric rows carried. Only OPEN numbers are the Gosavi INSTRUCTOR HANDOUT newsvendor values: MDH=15, six-sigma=31.53, newsvendor=26.96 (NOT a peer-reviewed number).

**Reproduced value (this audit):** vendor_managed_inventory_newsvendor_worked_case_summary(): mean_demand_rate=0.375, cycle_demand_mean=15.0, cycle_demand_variance=30.3646, MDH=15.0, six_sigma=31.53122, newsvendor=26.99054 (within tol). Worked transition period_cost=6.0 matches tests.rs. Re-tuned low_penalty retailer_base_stock held-out mean=103.012347 — bit-identical to ledger.

**Rerun method / tolerance:** invman_rust.vendor_managed_inventory_newsvendor_worked_case_summary(); vendor_managed_inventory_step(...); plus benchmark_reduced_single_retailer.tune_retailer_base_stock/heuristic_held_out_samples for low_penalty. Exact DP NOT exposed to Python, NOT re-run.

**Reproduce command(s):**

```bash
python -c "import invman_rust as m; print(m.vendor_managed_inventory_newsvendor_worked_case_summary())"
python -c "import invman_rust as m; print(m.vendor_managed_inventory_step(dc_on_hand=4,retailer_on_hand=1,retailer_pipeline=1,shipment_quantity=2,realized_demand=3,dc_replenishment_quantity=2,dc_capacity=5,shipment_cost_per_unit=0.4,dc_holding_cost_per_unit=0.3,retailer_holding_cost_per_unit=0.6,stockout_cost_per_unit=4.0))"
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py --quick
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py --description audit --budget full --instance low_penalty --tree_leaf_type linear --tree_depth 3 --tree_temperature 0.1 --warm_start base_stock --seed 777
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
