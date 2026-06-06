# Ameliorating inventory — benchmark card

**One-line MDP:** state = inventory-by-age vector `[I^1..I^A]` augmented with the realized stochastic purchase price `P_t`; action = scalar purchase volume `a^P_t ∈ [0, maxInventory]` that enters the youngest age class; one-period cost = NEGATIVE profit `ρ_t = R_t − P_t·a^P_t − h·Σ_a I^a_t` (realized sales revenue minus purchase cost minus holding); objective = maximize long-run AVERAGE profit `G = lim_{T→∞}(1/T)E[Σ ρ_t]`.

**Status:** `faithful_unverified` (the env is structurally faithful to Pahr & Grunow 2025; the published perfect-information LP bound was NOT re-run this audit — no Python binding — while the learned-policy env path WAS re-run). **Paper:** §"Ameliorating inventory" of `learning_inventory_control_policies_es.tex` (line 3160; formulation §3163, policy §3284, results §3336).

## Problem formulation

Single-item, periodic-review ameliorating inventory where the product *improves* with age (purchased young/cheap, sold older/more valuable), from Pahr & Grunow (2025), "The Value of Blending — Managing Ameliorating Inventory Using Deep Reinforcement Learning", Production and Operations Management 35(5), DOI 10.1177/10591478251387795.

- **Timing / state.** Observe inventory by age class `I_t = [I^1..I^A]` and the realized stochastic purchase price `P_t` (truncated-Normal, companion mean 200, std 50, truncated ±70).
- **Action.** Choose a scalar purchase volume `a^P_t ∈ [0, maxInventory]`. It enters the youngest age class: `I^1 ← a^P_t`.
- **Transition.** A per-period **blending LP** issues quantities `x^a_t` from the age classes to meet per-product demand, subject to per-product mean-age targets, blending / no-blending rules, and a processing capacity; sold quantities leave inventory. Surviving stock ages by one class; an age-dependent stochastic Beta decay (mean `decay_mean[a]`) plus a constant multiplicative evaporation `(1−evaporation)^(a+1)` removes a fraction of each class. The "three-part" purchase/produce/issue decision of the paper is reduced here to the structural purchase head — production is derived from the env's blending LP.
- **One-period profit.** `ρ_t = R_t − P_t·a^P_t − h·Σ_a I^a_t`, with `R_t` = realized sales revenue (issued quantities valued at the stochastic correlated sales prices) and `h` = per-unit holding cost. The faithful env reward in `average_profit_blending_env.rs` is `revenue − purchase_cost − holding + decay_salvage − outdating`; step ordering matches the companion `step_continuous_issuance_lp`.
- **Objective.** Maximize long-run average profit `G(θ) = lim_{T→∞}(1/T)E[Σ_{t=1}^T ρ_t]`, estimated by simulation over shared price/demand-path seeds on a long post-warm-up window.

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `pahr_grunow2025_spirits_0001` | A:10, products:3, regime:no_blending, objective:average_profit, maxInventory:50, role:PRIMARY verification anchor | target ages [2,4,6], evaporation 0.03, holding 2.5, blending OFF, LP bound 1991.9344293376805 | **true** (in `references.rs` `PRIMARY_REFERENCE_INSTANCE`) |
| `pahr_grunow2025_port_wine` | A:25, products:2, regime:blending_enabled, objective:average_profit, maxInventory:50, role:SECONDARY verification anchor | target ages [9,19], evaporation 0.02, holding 1.0, blending ON, LP bound 2444.8010643781136 | **true** (in `references.rs` `PORT_WINE_REFERENCE_INSTANCE`) |
| `spirits_0002` (blending ON) | A:10, products:3, regime:blending_enabled, maxInventory:50 | LP bound 1991.9344293376805 | **absent** — own re-solve test only, NOT in `references.rs` `REFERENCE_INSTANCES` catalog |
| `spirits_1002` (capacity-constrained) | A:10, products:3, regime:blending_enabled, maxInventory:30 | per-age capacity lowered 50→30, LP bound 1663.8888177082856 | **absent** — own re-solve test only, not in catalog |

## Baselines

- **Heuristics:** best tuned **order-up-to purchase "gate"** (purchase grid 2..24), plus the keep/discard comparator. (Reduced-model only, Rust-only: `newsvendor_purchase`, `two_dimensional_order_up_to`, and the reduced-model `finite_horizon_dp`.)
- **Exact / optimal:** none. The only exact solver is the **perfect-information steady-state LP** (`perfect_information_lp.rs::solve_upper_bound`, in-crate `microlp` simplex), which produces an **UPPER BOUND** on average profit (`max_reward`), NOT an achievable optimum. It is Rust-only with NO Python binding. Bounds: spirits_0001 = 1991.9344293376805, port_wine = 2444.8010643781136, spirits_1002 = 1663.8888177082856, spirits_0002 = 1991.9344293376805.
- **Published comparators (CONTEXT only — cross-protocol):** Pahr & Grunow (2025) report a deep-RL agent within ~3.5% of the bound using the FULL 3-part action (purchase + production targets + per-age issuance). This is NOT reproduced and is NOT comparable to this repo's single-purchase-action gap — context only, never "beaten."

## Verification

- **Published number:** perfect-information LP bounds — spirits_0001 = 1991.9344293376805; port_wine = 2444.8010643781136; spirits_1002 = 1663.8888177082856; spirits_0002 = 1991.9344293376805.
- **Re-run reproduced (debt CLOSED 2026-06-06):** the perfect-information LP bound is now exposed as the Python binding `ameliorating_inventory_perfect_info_lp_bound_summary(reference_name)` and **re-runs in <1 s**, reproducing the companion (Pahr–Grunow 2025 companion code) anchors to ~1e-8: spirits_0001 = **1991.9344293931** (companion 1991.9344293377), port_wine = **2444.801** (companion 2444.801). The learned-policy env path was also re-run: spirits_0001 smoke = 77.96 ± 0.74 vs gate 20.07 ± 0.95; full-budget = 115.07 ± 0.44 vs gate 20.91.
- **Verdict:** the **LP bound is now verified-by-rerun against the companion code** (reference-impl, not a peer-reviewed paper table); the **trainable env remains `faithful_unverified`** (no published *achieved cost* is reproduced by it). The LP is an UPPER BOUND, not an achievable optimum, so it anchors a *gap*, never a "beat."

## Results (learned policy)

All learned-vs-gate results below are reported by the manifest as **`single_seed`** and **`at_risk = true`** — they are **single-seed, NOT yet seed-robust** (the user's seed-robust standard is mean ± std over ≥5 optimizer seeds; the `±` figures here are paired-CRN eval SEM within one optimizer seed, not across optimizer seeds).

- spirits_0001: learned price-reactive purchase soft tree **115.07 ± 0.44** vs best tuned order-up-to gate **20.91** (+450% of gate); smoke reproduced 77.96 vs 20.07. **single-seed, NOT yet seed-robust.**
- port_wine: learned **505.78 ± 0.59** vs gate **133.78** (+278% of gate). **single-seed, NOT yet seed-robust.**
- spirits_1002 (capacity-constrained): learned **130.49 ± 0.50** vs gate **20.91** (+524% of gate); gap to bound 92.2%. **single-seed, NOT yet seed-robust.**
- **Gap to the perfect-information LP UPPER BOUND remains large and is reported as a gap, never "beaten":** 94.2% (spirits_0001), 79.3% (port_wine). This gap is structural — a single purchase action vs the bound's full 3-part LP issuance from inventory held at every age up to capacity, with the env charging the full purchase price (~200/unit) every period. It is NOT comparable to Pahr & Grunow's ~3.5% DRL gap (which uses the full 3-part action). (This last claim is the one row the manifest marks `at_risk = false`.) port_wine's tighter gap is the value of blending (issuance across target ages [9,19]).

## Reproduce

```bash
# Learned-policy env path (smoke; reproduced this audit). Optimizer seed 20250604.
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py --instance spirits_0001 --budget smoke --seed 20250604

# Full budget, port_wine
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python /home/nima/code/ml/invman/scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py --instance port_wine --budget full --seed 20250604

# Repo-native instance benchmark
python /home/nima/code/ml/invman/scripts/ameliorating_inventory/benchmark_repo_native_instance.py

# LP upper-bound re-solve (Rust-only; NOT run this audit — requires cargo compile)
cargo test -p invman_rust --lib problems::ameliorating_inventory::tests::verification -- --nocapture

# Inspect a committed learned result
python3 -c "import json;print(json.load(open('/home/nima/code/ml/invman/outputs/autoresearch/ameliorating_inventory_average_profit_autoresearch/spirits_0001_d1_oblique_full.json'))['learned'])"
```

## Pointers & caveats

- **code:** `src/problems/ameliorating_inventory/` — canonical faithful env `average_profit_blending_env.rs`; issuance LP `issuance_blending_lp.rs`; perfect-information upper-bound LP `perfect_information_lp.rs`; dataset loader `lp_dataset_loader.rs`; references/anchors `references.rs`; executing literature verification `tests/verification.rs`; checked-in companion datasets `practical/datasets/`. Reduced-model (NOT the verification target): `env.rs`, `issuance.rs`, `rollout.rs`, `heuristics/`, `finite_horizon_dp.rs`, `bindings.rs`, `demand.rs`, `literature/`, `verification/tests.rs`.
- **scripts:** `scripts/ameliorating_inventory/` (`autoresearch_ameliorating_inventory_average_profit.py`, `benchmark_repo_native_instance.py`, `RESULTS_FULL_BUDGET.md`, `PAPER_SECTION_DRAFT.md`).
- **autoresearch:** `autoresearch/program_ameliorating_inventory.md`.
- **binding:** the learned path uses `ameliorating_inventory_average_profit_soft_tree_population_rollout` (in `bindings.rs`), which targets the FAITHFUL `average_profit_blending_env.rs` (NOT the reduced discounted-cost `env.rs`), rolls out `step_state` under paired CRN, and returns long-run average profit. Policy = depth-1 oblique soft tree with a single purchase head and a linear leaf (price-reactive order-up-to), CMA-ES warm-started at order-up-to `a^P = softplus(S − Σ I^a)`.
- **CAVEAT — cross-protocol comparator.** Pahr & Grunow's ~3.5% DRL gap uses the full 3-part action; it is CONTEXT only and not a like-for-like comparison to this repo's single-purchase head. The repo never claims to match or beat it.
- **CAVEAT — bound is an upper bound.** The perfect-information LP is an UPPER BOUND, not an achievable optimum; results anchor a *gap to bound*, never a "beat."
- **CAVEAT — all learned-vs-gate results are single-seed (`at_risk`).** They are NOT yet seed-robust under the repo's ≥5-optimizer-seed mean±std standard; the reported `±` is within-seed paired-CRN eval SEM.
- **CAVEAT — verification debt.** No Python binding for the LP bound; it was not executed this audit. Closing the gap-to-bound (toward the paper's ~3.5%) is a recorded follow-up requiring a full 3-part action head (purchase + production targets + per-age issuance), new Rust rollout code, and a rebuild of `invman_rust` — out of scope for the prior run.
- **NOTE — existing README.md.** `src/problems/ameliorating_inventory/README.md` (lines 10-12) asserts `literature-verified: TRUE`. This is honest about the in-crate LP re-solve test being a genuine reproduction (gap < 1e-7) *when run*, but it CONTRADICTS the audit ledger, which records `faithful_unverified` because that Rust-only test was NOT executed this audit (no Python binding; read not run). Per instructions this README was left unedited; treat the ledger's `faithful_unverified` as the authoritative status.
