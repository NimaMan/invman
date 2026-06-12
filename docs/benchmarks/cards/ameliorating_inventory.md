# Benchmark card — `ameliorating_inventory`

**Subfamily:** Pahr & Grunow 2025 'The Value of Blending' (average profit)

**Difficulty:** `medium` — Age-indexed inventory across A=10..25 age classes with a 3-part action (purchase / keep-discard / blend) in the full model; the only solver is a perfect-information steady-state LP UPPER BOUND (bound_gap, never an achievable optimum, no Python binding), so it is gap-to-bound rather than true-optimum benchmarking.

**Verification tier:** `reference` (re-runs a companion / closed-form / reduced-module number, or a published action)

**Tier note:** Split: the perfect-info LP bound is verified_rerun vs the Pahr-Grunow 2025 companion code (~1e-8, reference-grade) -> headline tier = reference; the TRAINABLE env is faithful_unverified (no published achieved cost; the LP is an upper bound, not a reproduced optimum).

> Status (manifest, verbatim): faithful_unverified (LP bound NOT re-run this audit — no Python binding; learned-policy env path WAS re-run)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| pahr_grunow2025_spirits_0001 | true | A:10, products:3, regime:no_blending, objective:average_profit, maxInventory:50, role:PRIMARY_verification_anchor, LP_bound:1991.9344293376805 |
| pahr_grunow2025_port_wine | true | A:25, products:2, regime:blending_enabled, objective:average_profit, maxInventory:50, role:SECONDARY_verification_anchor, LP_bound:2444.8010643781136 |
| spirits_0002 (blending ON) | absent (own re-solve test but NOT in references.rs REFERENCE_INSTANCES) | A:10, products:3, regime:blending_enabled, maxInventory:50, LP_bound:1991.9344293376805 |
| spirits_1002 (capacity-constrained) | absent (own re-solve test only; not in catalog) | A:10, products:3, regime:blending_enabled, maxInventory:30, LP_bound:1663.8888177082856 |

## Baselines

**Heuristics**
- best tuned order-up-to purchase 'gate' (grid 2..24); the keep/discard comparator
- (reduced-model only) newsvendor_purchase and two_dimensional_order_up_to + reduced-model finite_horizon_dp (Rust-only)

**Exact solver / bound**

perfect-information steady-state LP (perfect_information_lp.rs::solve_upper_bound, microlp simplex) — published max_reward UPPER BOUND on average profit, NOT an achievable optimum. Rust-only, NO Python binding.

**Published rows**
- spirits_0001 LP bound = 1991.9344293376805
- port_wine LP bound = 2444.8010643781136
- spirits_1002 LP bound = 1663.8888177082856
- spirits_0002 LP bound = 1991.9344293376805
- Pahr & Grunow report deep-RL within ~3.5% of bound using FULL 3-part action — NOT reproduced and NOT comparable to this repo's single-purchase-action gap

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `single_seed` | True | no | Learned price-reactive purchase soft tree beats best tuned order-up-to gate: spirits_0001 115.07+/-0.44 vs 20.91 (+450%); smoke reproduced 77.96 vs 20.07. |
| `single_seed` | True | no | port_wine learned 505.78+/-0.59 vs gate 133.78 (+278%). |
| `single_seed` | True | no | spirits_1002 (capacity) learned 130.49+/-0.50 vs gate 20.91 (+524%); gap to bound 92.2%. |
| `single_seed` | False | no | Gap to perfect-info LP UPPER BOUND remains large (94.2% spirits_0001, 79.3% port_wine), reported as a gap, never 'beaten'. NOT comparable to Pahr & Grunow ~3.5% DRL gap. |

## How to reproduce & compare

**Expected (published) value:** perfect-information LP bounds: spirits_0001 = 1991.9344293376805; port_wine = 2444.8010643781136; spirits_1002 = 1663.8888177082856; spirits_0002 = 1991.9344293376805

**Reproduced value (this audit):** LP bound NOT re-run (no Python binding; needs cargo compile). In-crate tests/verification.rs RE-SOLVES the LP and asserts within 1e-3 (genuine reproduction, read not executed). Learned-policy env re-run: spirits_0001 smoke = 77.96+/-0.74 vs gate 20.07+/-0.95; full-budget = 115.07+/-0.44 vs gate 20.91.

**Rerun method / tolerance:** RAYON_NUM_THREADS=4 python scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py --instance spirits_0001 --budget smoke --seed 20250604 (uses ameliorating_inventory_average_profit_soft_tree_population_rollout). LP bound re-solve requires cargo test (NOT run).

**Reproduce command(s):**

```bash
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py --instance spirits_0001 --budget smoke --seed 20250604
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py --instance port_wine --budget full --seed 20250604
python /home/nima/code/ml/invman/scripts/ameliorating_inventory/benchmark_repo_native_instance.py
cargo test -p invman_rust --lib problems::ameliorating_inventory::tests::verification -- --nocapture
python3 -c "import json;print(json.load(open('/home/nima/code/ml/invman/outputs/autoresearch/ameliorating_inventory_average_profit_autoresearch/spirits_0001_d1_oblique_full.json'))['learned'])"
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._

