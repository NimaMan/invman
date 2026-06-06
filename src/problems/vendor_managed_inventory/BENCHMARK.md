# vendor_managed_inventory — benchmark card

**One-line MDP:** state = (period, DC on-hand, retailer on-hand, retailer pipeline); action = integer shipment quantity from the vendor's DC into the retailer's consignment stock (clipped to `min(max_shipment, dc_on_hand)`); one-period cost = shipment + DC-holding + retailer-holding + lost-sales penalty; objective = minimize discounted ($\gamma=0.99$) finite-horizon expected cost over the episode (lower is better).

**Status:** `faithful_unverified` — NO peer-reviewed published number is reproduced. The trainable env is structurally faithful and only repo-native/handout anchors re-ran. (Per VERIFICATION_LEDGER.md, audited 2026-06-06.)  **Paper:** none — `vendor_managed_inventory` is NOT written up in `learning_inventory_control_policies_es.tex`; it is a repo problem family outside the paper's ~10 validated environments (grep of the .tex for vendor/consignment/Sui returns 0 hits).

## Problem formulation

The benchmarked, trainable env is the **reduced single-retailer finite-horizon VMI slice** (`env::step_state`), the only VMI env exposed to Python via the `invman_rust.vendor_managed_inventory_*` bindings. A supplier-owned distribution center (DC) replenishes one downstream retailer's consignment stock under a one-period shipment lead time. Timing within a period (from `env.rs::step_state`, lines 133–200):

1. **State** $(t,\, I^{DC}_t,\, I^{R}_t,\, P^{R}_t)$ — period index, DC on-hand, retailer on-hand, retailer in-transit pipeline.
2. **Action** — shipment quantity $q_t$, clipped to $\min(\text{max\_shipment},\, I^{DC}_t)$ (the DC cannot ship what it does not hold).
3. **Transition** — pipeline arrives first: $\text{available} = I^{R}_t + P^{R}_t$; demand $D_t$ realizes; $\text{sales}=\min(\text{available}, D_t)$; lost sales $= D_t - \text{sales}$ (lost-sales regime, no backorder); next retailer on-hand $= \text{available}-\text{sales}$; next pipeline $= q_t$. DC: $I^{DC}_t - q_t$, then deterministic upstream replenishment $\min(\text{dc\_replenishment\_quantity},\, \text{dc\_capacity} - (I^{DC}_t - q_t))$ refills toward `dc_capacity`.
4. **One-period cost** $= s\,q_t + h^{DC}\,I^{DC}_{t+1} + h^{R}\,I^{R}_{t+1} + p\cdot(\text{lost sales})$ where $s$=shipment cost/unit, $h^{DC}/h^{R}$=holding cost/unit, $p$=stockout cost/unit. Reward $=-\text{period\_cost}$.
5. **Objective** — minimize $\sum_t \gamma^t\,\text{period\_cost}_t$ with $\gamma=0.99$ over a 24-period horizon (terminal salvage credit available via `terminal_salvage_credit`).

`env.rs` also defines a **full continuous-time, 10-retailer / 2-product truck-dispatch simulator** (`PaperVendorManagedInventoryModel`, `step_paper_state`) that mirrors the paper's structure (compound-Poisson demand, random route cycle times, truck-capacity dispatch, DC `(Q,R)` rule). It is faithful-but-unreproduced and is NOT the benchmark anchor — its parameter rows are a repo-constructed interpretation, not a transcribed published table.

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| PRIMARY_REFERENCE_INSTANCE / `sui_gosavi_lin_2010_style_single_retailer` | lost_sales; single_retailer; periods 24; demand Poisson(2.5); repo-chosen, no published anchor | stockout 5.0, dc_capacity 10, max_shipment 5, $h^{DC}$=0.25, $h^{R}$=0.6, ship 0.4, $\gamma$=0.99 | **false** (`SUI_GOSAVI_LIN_2010_REFERENCE.literature_verified=false`; repo-chosen params, not a published table) |
| low_penalty | lost_sales; perturbation of primary | stockout 2.0 | false |
| high_penalty | lost_sales; perturbation; widest learned loss | stockout 9.0 | false |
| low_demand | lost_sales; perturbation | demand Poisson(1.5) | false |
| high_demand | lost_sales; perturbation | demand Poisson(3.5) | false |
| VERIFICATION_PROBLEM_INSTANCE (exact-DP verifier) | lost_sales; periods 5; discrete demand support {0,1,2,3}; small enough for exact DP | $\gamma$=0.99, dc_capacity 5, max_shipment 4 | false (repo-native self-consistency verifier; NOT a published number) |
| SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE (newsvendor handout) | compound_poisson; single retailer/product; cycle time {30,40,50}; newsvendor order-up-to | $\lambda$=0.25, demand~U(1,2), $h$=0.06, $p$=4.0 | false (instructor TEACHING HANDOUT, NOT peer-reviewed) |
| SUI_GOSAVI_LIN_2010_CASE_DEFINITIONS (8-case truck-dispatch) | truck_dispatch; 10 retailers × 2 products; continuous time; structural interpretation | 8 factorial cases over penalty/holding/demand-rate levels | false (repo-constructed interpretation; does NOT reproduce the paywalled table) |

## Baselines

- **Heuristics** (tuned on TRAIN seeds, scored on a disjoint held-out CRN seed block):
  - `retailer_base_stock` — best base-stock level by grid search.
  - `dc_reserve_base_stock` — best (base-stock level × DC reserve) by grid search. On this slice the two are nearly identical; the keep/discard target is `min(retailer_base_stock, dc_reserve_base_stock)` per instance.
  - Truck-dispatch family only: `paper_mean_demand` (MDH order-up-to) and `paper_newsvendor` (newsvendor order-up-to + truck allocation).
- **Exact / optimal:** `finite_horizon_dp::solve_optimal_policy` exists in Rust (the TRUE finite-horizon optimum on the reduced discrete-demand instance) and is used by an in-crate dominance test — but it is **NOT exposed as a Python binding**, so it is NOT used as the benchmark optimality ceiling. The reported gaps are learned-vs-strongest-heuristic, not learned-vs-optimum.
- **Published comparators (CONTEXT only):** the peer-reviewed Sui/Gosavi/Lin (2010) RL-vs-newsvendor profit table is **paywalled** and not carried. The only OPEN concrete numbers are from the Gosavi instructor HANDOUT (newsvendor order-up-to: MDH $S$=15, six-sigma $S$=31.53, newsvendor $S$=26.96) — a teaching handout, NOT a peer-reviewed result, and NOT a "beats" target.

## Verification

- **Published number:** NONE peer-reviewed. The paper results table is paywalled; no numeric rows are carried. The only OPEN numbers are the Gosavi INSTRUCTOR HANDOUT newsvendor order-up-to values: MDH=15, six-sigma=31.53, newsvendor=26.96.
- **Re-run reproduced:** `vendor_managed_inventory_newsvendor_worked_case_summary()` returns mean_demand_rate=0.375, cycle_demand_mean=15.0, cycle_demand_variance=30.364583, **MDH=15.0, six_sigma=31.53122, newsvendor=26.99054** (within tolerance of the handout's 15 / 31.53 / 26.96) via `python -c "import invman_rust as m; print(m.vendor_managed_inventory_newsvendor_worked_case_summary())"` (re-run confirmed this audit). The reduced-slice env step reproduces `period_cost=6.0` against `tests.rs`, and the re-tuned `low_penalty` retailer_base_stock held-out mean is 103.012347 (bit-identical to the ledger). **Verdict: faithful_unverified.**
- **Debt / caveat (stated plainly):** a handout is explicitly NOT literature verification per the repo rule, and the exact finite-horizon DP optimum is NOT exposed to Python, so it was NOT re-run as a ceiling. There is therefore **no public per-instance number to target** for this family. The truck-dispatch case definitions are a repo-constructed structural interpretation whose profit rows do not reproduce the paywalled table (an earlier audit found reproduced case-1 newsvendor profit ~16.4 vs a figure-read ~15.41 — not close enough to anchor).

## Results (learned policy)

All learned results below are **single-seed, NOT yet seed-robust** (the manifest marks them `at_risk`). The repo's reporting standard requires mean ± std over ≥5 optimizer seeds; these are single CMA-ES configs, not multi-seed, so they must be read as suggestive, not established wins.

- **README baseline** (constant leaf, no warm-start, depth 2, oblique, temp 0.1; full budget: 64 train seeds, 200 iters, 32 held-out heuristic seeds × 1500 reps, 4000 soft-tree held-out seeds; all SEMs < 0.4) — the learned soft tree **LOSES on 4/5** instances: primary −1.76% (115.75 → 117.80), low_penalty −0.16% (103.01 → 103.18), high_penalty −2.40% (124.34 → 127.33), high_demand −0.91% (119.54 → 120.63); marginally wins low_demand +0.10% (101.6x → 101.50). (`seed_reporting=single_seed`, NOT at_risk-flagged because it is the published-style baseline.)
- **Autoresearch (linear leaf + base-stock CMA warm-start), single config:** low_penalty FLIPS to a clean win — learned 102.69 vs tuned heuristic 103.01, gap **−0.31%** (margin > its SEM 0.11). **single-seed, NOT yet seed-robust** (`at_risk=true`).
- **Same lever, other instances, single config:** primary closed −1.76% → +0.05% (statistical tie, inside SEM 0.19); high_penalty −2.40% → +0.30% (loss closed ~8×, gap ~ SEM 0.27); high_demand −0.91% → +1.12% (single config, not best-tuned). All **single-seed, NOT yet seed-robust** (`at_risk=true`). Net: no robust, multi-seed-established win exists on this convex single-stage slice; the tuned base-stock heuristic is essentially optimal.
- **Honest best-of floor (training-path audit 2026-06-06):** `autoresearch_vendor_managed_inventory.py` gained an ADDITIVE `--deploy_endpoint {floor,xbest,xfavorite}` flag (default `floor`). The local CMA-ES train now also returns `es.result.xfavorite` (the distribution mean — the local-train analog of `es_mp.train`'s `es.current_param()`); the floor deploys the best-of {xbest, xfavorite, base-stock warm-start anchor} on the held-out soft-tree block. It is DOWNSIDE-SAFE: `--deploy_endpoint xbest` reproduces the historical xbest deployment bit-identically, and the floor never deploys worse than xbest. **Seed-robust quantification, high_penalty (screening budget, 5 optimizer seeds {11,22,33,44,55}, linear leaf + base-stock warm-start):** xbest seed-mean **126.42 ± 0.28** vs floored seed-mean **126.30 ± 0.47** vs same-protocol gate (tuned `retailer_base_stock`) **124.25**. The floor deployed xfavorite in 1/5 seeds (seed=11: 125.49 < xbest 126.09) and reproduced xbest in 4/5; the warm-start anchor (194.99) was always dominated. **Verdict unchanged: still a LOSS** — floored gap vs gate **+1.65%** vs xbest **+1.75%** (the floor narrows the loss ~0.1pp but does not flip it to parity/win). Consistent with the convex single-stage slice where xbest does not materially overfit, so the floor's value here is bounded.

## Reproduce

```bash
# Newsvendor worked-case summary (handout anchor; MDH/six-sigma/newsvendor order-up-to)
python -c "import invman_rust as m; print(m.vendor_managed_inventory_newsvendor_worked_case_summary())"

# Single reduced-slice env step (period_cost should be 6.0 for these inputs)
python -c "import invman_rust as m; print(m.vendor_managed_inventory_step(dc_on_hand=4,retailer_on_hand=1,retailer_pipeline=1,shipment_quantity=2,realized_demand=3,dc_replenishment_quantity=2,dc_capacity=5,shipment_cost_per_unit=0.4,dc_holding_cost_per_unit=0.3,retailer_holding_cost_per_unit=0.6,stockout_cost_per_unit=4.0))"

# Reduced single-retailer benchmark (heuristic tuning + learned soft tree, held-out CRN); HARD CPU cap
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py --quick

# Autoresearch: linear-leaf + base-stock warm-start on the widest-loss instance
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py --description audit --budget full --instance low_penalty --tree_leaf_type linear --tree_depth 3 --tree_temperature 0.1 --warm_start base_stock --seed 777

# Honest best-of floor (deploy best-of {xbest, xfavorite, warm-start anchor}); --deploy_endpoint xbest reproduces the historical xbest deployment exactly
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py --description floor --budget screening --instance high_penalty --tree_leaf_type linear --warm_start base_stock --deploy_endpoint floor --seed 11
```

## Pointers & caveats

- code: `src/problems/vendor_managed_inventory/env.rs` (reduced-slice `step_state` + truck-dispatch `step_paper_state`), `literature/references.rs` (instances + honesty flags, all `literature_verified=false`), `verification/newsvendor_case.rs` + `verification/tests.rs` (handout reproduction + DP dominance/drift guard), `finite_horizon_dp.rs` (exact optimum, Rust-only), `bindings.rs` (`vendor_managed_inventory_*` Python bindings).
- scripts: `scripts/vendor_managed_inventory/` (`benchmark_reduced_single_retailer.py`, `autoresearch_vendor_managed_inventory.py`).
- autoresearch: `autoresearch/program_vendor_managed_inventory.md`.
- **Honest caveats:**
  - **No published anchor.** The peer-reviewed paper table is paywalled; the only OPEN numbers are an instructor handout (NOT literature verification). Status is `faithful_unverified` with `no_published_number`.
  - **Not in the ES paper.** Unlike the paper's validated environments, `vendor_managed_inventory` has no `§` in `learning_inventory_control_policies_es.tex`.
  - **No optimality ceiling exposed.** `finite_horizon_dp::solve_optimal_policy` (true optimum) is Rust-only; reported gaps are learned-vs-heuristic, not learned-vs-optimum. Exposing it (Rust rebuild + `bindings.rs` edit) is the top next step.
  - **All learned "wins"/"ties" are single-seed.** The low_penalty −0.31% flip and the primary/high_penalty/high_demand closures are single CMA-ES configs (`at_risk=true`); none is yet reported as mean ± std over ≥5 seeds per the repo seed-robust standard.
  - **Truck-dispatch family is interpretive.** The 10-retailer / 2-product `PaperVendorManagedInventoryModel` and 8 case definitions are a repo-constructed structural interpretation, not a transcribed published table, and do not reproduce the paywalled profit rows.
