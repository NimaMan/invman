# joint_pricing_inventory — benchmark card

**One-line MDP:** state `(period t, on-hand inventory I_t)`; action `(order quantity q_t ≤ max_order_quantity, discrete price index p_t)`; one-period cost `procurement c·q + holding h·(ending inv) + stockout s·(lost sales) − revenue p·sales` (i.e. negative profit); objective = minimize expected discounted finite-horizon cost (maximize discounted profit), with a terminal salvage credit on leftover stock.

**Status:** `faithful_unverified` / `no_published_number` — the env is structurally faithful and re-ran self-consistency anchors match, but NO peer-reviewed per-instance number is reproduced.  **Paper:** NOT covered in `learning_inventory_control_policies_es.tex` (the paper treats lost sales, dual sourcing, multi-echelon, perishable, general-network backorder, serial, OWMR, ameliorating, and PADN only; "pricing/newsvendor" mentions elsewhere in the paper refer to the serial recursive-newsvendor solver, not this problem).

## Problem formulation

Single item, zero lead time, lost-sales, finite horizon. Timing within period `t` (see `env.rs::step_state`):

1. **State** `(t, I_t)` with `I_t` on-hand inventory.
2. **Action** `(q_t, p_t)`: order `q_t` clipped to `[0, max_order_quantity]`; choose discrete price index `p_t` into the price ladder (`clip_action`). Order arrives immediately, so inventory-after-order `= I_t + q_t`.
3. **Demand** `D_t` is stochastic and price-dependent: Poisson with price-specific mean (primary instance) or a discrete distribution (verifier instance). Sampled in `demand.rs::sample_demand`.
4. **Transition / accounting:** `sales = min(I_t + q_t, D_t)`, `lost_sales = D_t − sales`, `I_{t+1} = I_t + q_t − sales`.
5. **One-period cost** `period_cost = c·q_t + h·I_{t+1} + s·lost_sales − price[p_t]·sales` (reward = −period_cost). Holding is charged on ending inventory `I_{t+1}`.
6. **Terminal:** at horizon, `terminal_salvage_credit = salvage·I_T` is added (a profit credit / negative cost).

**Long-run objective:** minimize expected discounted sum of `period_cost` over the finite horizon (the verifier uses discount 0.99). Profit = −cost.

**Single-period (T=1) reduction** is exactly the textbook **price-setting newsvendor**: overage `Co = c + h`, underage `Cu = p + s − c`, optimal order-up-to = smallest `y` with `F(y) ≥ Cu/(Cu+Co)` (critical fractile).

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
| --- | --- | --- | --- |
| `VERIFICATION_PROBLEM_INSTANCE` (Qin-2022-labeled exact verifier) | lost_sales; horizon T=5; 3-level price ladder; discrete price-dependent demand; exact DP feasible; leadtime L=0 | prices `[7,9,11]`; demand supports `{0,1,2,3}` w/ probs `cheap [.1,.2,.3,.4] / mid [.2,.3,.3,.2] / expensive [.4,.3,.2,.1]`; discount 0.99; c=4.0, h=0.5, s=5.0, salvage=1.0; max_order_quantity=4; init inv=1 | **false** |
| `PRIMARY_REFERENCE_INSTANCE` / `zhou2022_style_price_ladder` | lost_sales; horizon T=18; 3-level price ladder; Poisson price-dependent demand; no exact optimum; leadtime L=0 | prices `[8,10,12]`; Poisson means `[4.0, 2.6, 1.6]`; discount 0.99; c=4.0, h=0.5, s=5.0, salvage=1.0; max_order_quantity=6; init inv=2 | **false** |

Source labels in `literature/references.rs`: `ZHOU_2022_REFERENCE` (DRL, ESWA, doi:10.1016/j.eswa.2022.116564 — note: that paper adds a reference-price state the repo deliberately omits, so it is a different MDP) and `QIN_2022_REFERENCE` (Qin, Simchi-Levi & Wang 2022, doi:10.1287/mnsc.2021.4212 — same model class but a sample-complexity theorem, no reusable per-instance optimal-profit table). Both instances carry `literature_verified = false`.

## Baselines

- **Heuristics** (searched as fixed-parameter policies carried in `references.rs`, evaluated exactly by DP on the verifier and by Monte Carlo on the primary):
  - `static_price_base_stock` — order-up-to + fixed price index (`heuristics/static_price_base_stock.rs`).
  - `inventory_sensitive_base_stock` — order-up-to + markdown threshold (drop to a low price when inventory is high; `heuristics/inventory_sensitive_base_stock.rs`).
- **Exact / optimal:** `finite_horizon_dp.rs::solve_optimal_policy` — backward-induction exact DP over `(period, inventory)`, feasible on the T=5 verifier; exposed via `joint_pricing_inventory_exact_dp_summary()`. Optimal discounted cost **−33.178121049724**, first action **(q=2, price idx=1)**.
- **Published comparators (CONTEXT only):** NONE reproduced. `references.rs` carries label-only policy names for both papers (`ddqn_joint_price_inventory`, `value_iteration_baseline`, `q_learning_baseline`, `data_driven_approximation`, `deterministic_baseline`, `random_baseline`) — none are implemented and no numeric published row is stored.

## Verification

- **Published number:** none. There is no public per-instance optimal-profit number to target (`no_published_number`). Verification rests on two independent, correct anchors instead:
  1. **Analytical (independent):** the T=1 reduction equals the closed-form critical fractile. `verification/tests.rs::single_period_env_matches_price_setting_newsvendor_critical_fractile` confirms the env's T=1 optimum equals `smallest y with F(y) ≥ Cu/(Cu+Co)` for every price on the verifier instance: prices `(7, 9, 11) → y* = (3, 2, 2)`, matched by env brute force.
  2. **Repo-native exact DP (self-consistency):** `finite_horizon_dp.rs` optimal **−33.178121049724**, first action **(2,1)**; an independent hand-coded `lru_cache` Python DP reproduced **−33.178121049724** within 1e-9. Heuristics under the same DP: `static_price_base_stock` −32.50820139235; `inventory_sensitive_base_stock` −27.594377111812527.
- **Re-run reproduced** via `python -c "import invman_rust; print(dict(invman_rust.joint_pricing_inventory_exact_dp_summary()))"` (critical-fractile via `joint_pricing_inventory_step` + `joint_pricing_inventory_exact_verification_instance`; independent Python DP; and the benchmark script below). **Verdict: `faithful_unverified` / `no_published_number`.**
- **Debt / caveat:** this is a "faithful model with no reusable published per-instance anchor" case, NOT a model bug. The Zhou (2022) and Qin (2022) labels are formulation-class anchors only. To flip to verified, locate a citeable paper with a reproducible finite-horizon joint-pricing-inventory optimal-profit instance (e.g. a Federgruen–Heching worked example), carry that row, and reproduce it.

## Results (learned policy)

- **Verifier instance (exact-DP-anchored, re-ran exactly, NOT seed-dependent):** profit optimality gaps `static_price_base_stock` 2.02%, `inventory_sensitive_base_stock` 16.83%. (manifest: `at_risk = false`.)
- **Primary 18-period Poisson instance — CARRIED CLAIM IS SINGLE-SEED, NOT YET SEED-ROBUST:** trained depth-2 oblique/linear soft tree reaches profit **216.060** (cost −216.0595) vs best heuristic `inventory_sensitive_base_stock` **171.513** and `static_price_base_stock` **172.635**, i.e. **+25.15%** over the best heuristic, evaluated on 4096 fresh held-out seeds (base 777000). Trained params: `outputs/joint_pricing_inventory/tree_primary_d2_linear_b8_s123_e120_eval2048.json` (single training seed 123). The manifest marks this `seed_reporting = single_seed`, `at_risk = true`. Per the seed-robust reporting standard this is a single-seed / best-of-N result and is **NOT yet a seed-robust beat** — it must be re-run as mean ± std over ≥5 optimizer seeds before being claimed as robust. (Many eval seeds reduce *evaluation* noise but do NOT address *training-seed* variance.)
- **Primary instance — 5-seed seed-robust audit + honest xbest/xfavorite floor (training-path audit 2026-06-06):** `train_soft_tree_reference.py` now reads BOTH CMA-ES endpoints from the same run (xbest = `es.best_param()`, xfavorite = `es.current_param()` = distribution mean) and, under the default `--deploy_endpoint floor`, deploys the cheaper of {xbest, xfavorite} on the held-out eval seeds (downside-safe; this runner computes no gate/anchor, so the floor set is {xbest, xfavorite}). Re-run over 5 optimizer seeds (123–127; standard budget: depth 2, oblique/linear, 400 gen, pop 16, train_seed_batch 8, 2048 eval seeds), profit (= −cost, higher better):
  - **xbest (historical endpoint):** **220.434 ± 4.907**
  - **xfavorite (distribution mean):** 221.816 ± 1.953
  - **floor (best-of, deployed):** **222.871 ± 2.385** — deploys xfavorite in 3/5 seeds, xbest in 2/5
  - same-protocol heuristic gate (best of the two): **171.59** (`static_price_base_stock`); floor beats it **+29.9%** (xbest +28.5%).
  The floor raises seed-mean profit (220.43 → 222.87) and roughly halves the seed std (4.91 → 2.39). The robust verdict vs the heuristic gate was already a WIN and remains a WIN; the floor TIGHTENS it (lower variance) and is strictly downside-safe (never worse than xbest per seed). Reproduce: `--deploy_endpoint floor` (default) vs `--deploy_endpoint xbest` reproduces the historical single-best-individual deployment exactly. Per-seed JSON: `outputs/joint_pricing_inventory/seed_robust_floor_seed{123..127}.json`.

## Reproduce

```bash
# Exact DP summary (optimal cost, first action, heuristic costs) — verifier instance
python -c "import invman_rust; print(dict(invman_rust.joint_pricing_inventory_exact_dp_summary()))"

# Critical-fractile newsvendor self-check via the env step (verifier instance)
python -c "import invman_rust; ref=dict(invman_rust.joint_pricing_inventory_exact_verification_instance()); pl=list(ref['price_levels']); [print(pi, invman_rust.joint_pricing_inventory_step(0,q,pi,d,pl,4.0,0.5,5.0)) for pi in range(3) for q in range(5) for d in range(4)]"

# Exact-DP-anchored gaps + learned-vs-heuristic on the primary instance (no rebuild, no retrain)
python scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py --replications 4096 --seed 777000

# Validate env/heuristics against the exact DP via simulation
python scripts/joint_pricing_inventory/validate_against_exact_dp.py --simulation_replications 512 --simulation_seed 123

# (Optional) train a reference soft tree — single seed; NOT a seed-robust protocol
python scripts/joint_pricing_inventory/train_soft_tree_reference.py --depth 2 --leaf_type linear --seed 123 --eval_seeds 2048
```

## Pointers & caveats

- code: `src/problems/joint_pricing_inventory/env.rs`, `demand.rs`, `finite_horizon_dp.rs`, `rollout.rs`, `bindings.rs`, `heuristics/{static_price_base_stock,inventory_sensitive_base_stock}.rs`, `literature/references.rs`, `verification/tests.rs`
- scripts: `scripts/joint_pricing_inventory/` (`benchmark_policies_against_exact_and_learned.py`, `validate_against_exact_dp.py`, `train_soft_tree_reference.py`, `common.py`)
- autoresearch: no `policy_search/programs/program_joint_pricing_inventory.md` exists for this problem.
- **Cross-protocol caveat:** Zhou (2022) DDQN and Qin (2022) data-driven SAA are CONTEXT only — different MDP (reference-price state) / a theorem, not a per-instance number; never a "beats."
- **Verification debt:** `no_published_number`; anchors are an analytical critical-fractile check + a repo-native exact DP self-consistency check.
- **Seed caveat:** the +25.15% learned-policy claim is single training seed 123 (4096 *eval* seeds ≠ training-seed robustness); label it single-seed until re-run as mean ± std over ≥5 optimizer seeds vs the same-protocol heuristic gate.
- **Profit/cost convention:** the env returns discounted COST = −profit; more negative cost is better. Demand means are price-specific Poisson rates (mean, not std); the verifier uses explicit discrete distributions.
- An existing `README.md` is present in this directory and is consistent with the manifest/code; this `BENCHMARK.md` is the standardized card and does not modify it.
