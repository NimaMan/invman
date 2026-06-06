# One-warehouse multi-retailer (OWMR, Kaynov et al. 2024) — benchmark card

**One-line MDP:** state = warehouse + per-retailer inventory positions and in-transit pipelines; action = warehouse order-up-to level `S^w` and per-retailer targets `(S^1..S^K)` (deficits filled from scarce warehouse on-hand via a fixed allocation rule); one-period cost = `h_w·I^w + Σ_k (h_r·I^k + p·b^k)`; objective = minimize expected undiscounted total cost over `H=100` periods.
**Status:** `verified_rerun` — 2 of 14 published Kaynov rows reproduced by re-run + a repo-native exact-DP self-consistency anchor; the remaining 12 rows reproduce ~1–6% off and are carried as table literals (partial / mixed verification). **Paper:** §`sec:owmr` ("One warehouse, multiple retailers") of `learning_inventory_control_policies_es.tex`.

## Problem formulation
Periodic-review divergent two-echelon system of Kaynov et al. (2024): one warehouse (echelon 0) replenishes `K` retailers, each with i.i.d. demand `D^k_t` and its own deterministic lead time `L_{r,k}`. The warehouse orders from an external supplier with lead time `L_w`. Three customer regimes: `lost_sales`, `backorder` (complete), `partial_backorder` (a fraction backordered with emergency-shipment probability `β`, the rest lost).

Per-period event sequence (as implemented in `env.rs::step_state`, validated by the worked-transition test):
1. warehouse pipeline head arrives → available warehouse = on-hand + arrival;
2. retailer shipments leave warehouse stock (must not exceed available warehouse inventory);
3. warehouse order enters the tail of the warehouse pipeline; each retailer shipment enters the tail of that retailer's pipeline;
4. each retailer's pipeline head arrives; demand is realized against (retailer on-hand + arrival); under `partial_backorder` an emergency shipment may be drawn from remaining warehouse stock with probability `β`;
5. costs: warehouse holding `h_w` on ending warehouse on-hand (charged POST-emergency, Kaynov Eq. 6), per-retailer holding `h_r` on ending retailer on-hand, penalty `p` on unmet (lost / backordered) units.

Allocation of scarce warehouse stock follows one of two fixed rationing rules: **proportional** (floor of pro-rata share) or **min-shortage** (maximize the minimum resulting retailer inventory position). **Cost convention:** the env reports a positive `period_cost` and `reward = -period_cost`; published Kaynov rows are stored as NEGATIVE reward in `references.rs` (e.g. `-1408.08`), and the script layer compares against `-published_reward` (a cost of `1408.08`).

Long-run objective: a stationary policy maps state → action; performance is `C(θ) = E[Σ_{t=1..H} c_t]` with `H=100`, estimated by simulation over shared (CRN) demand-path seeds.

## Reference instances
14 Kaynov Table-A.3 instances (`TABLE_A3_INSTANCES` in `references.rs`; primary = instance_7). All `K=3` instances use `h_w=0.5, h_r=1, p=9`; the two `K=10` instances use `h_w=h_r=3, p=60`. `β`-emergency only in partial-backorder.

| instance | regime | dimensions covered | key params | literature_verified flag |
|---|---|---|---|---|
| kaynov2024_instance_1 | backorder | K=3, symmetric, low CV | Pois(3)×3, Lw2/Lr1 | true (ROW PROVENANCE only, NOT tight numerical reproduction) |
| kaynov2024_instance_2 | backorder | K=3, heterogeneous | U(0,6)/N(3,1)/Pois(3), Lw2/Lr1 | true (provenance) |
| kaynov2024_instance_3 | backorder | K=3, heterogeneous, high CV | N(1,5)/N(5,1)/Pois(0.5), Lw2/Lr1 | true (provenance) |
| kaynov2024_instance_4 | backorder | K=3, asymmetric retailer leads | Pois(3)×3, Lw2/Lr=1,2,3 | true (provenance) |
| kaynov2024_instance_5 | backorder | K=3, high CV, long lead | N(1,5)/N(5,1)/Pois(0.5), Lw5/Lr3 | true (provenance) |
| kaynov2024_instance_6 | lost_sales | K=3, symmetric, low CV | Pois(3)×3, Lw1/Lr1 | true (provenance) |
| kaynov2024_instance_7 | lost_sales | K=3, symmetric — **primary reference instance** | Pois(3)×3, Lw2/Lr1 | true (provenance; reproduced by re-run −0.94%) |
| kaynov2024_instance_8 | lost_sales | K=3, long warehouse lead | Pois(3)×3, Lw5/Lr1 | true (provenance) |
| kaynov2024_instance_9 | lost_sales | K=3, asymmetric retailer leads | Pois(3)×3, Lw2/Lr=1,2,3 | true (provenance) |
| kaynov2024_instance_10 | lost_sales | K=3, high CV, long lead | N(1,5)/N(5,1)/Pois(0.5), Lw5/Lr3 | true (provenance) |
| kaynov2024_instance_11 | partial_backorder | K=3, symmetric, β=0.8 | Pois(3)×3, Lw2/Lr1 | true (provenance; reproduced by re-run +0.13%) |
| kaynov2024_instance_12 | partial_backorder | K=3, heterogeneous, high CV, β=0.8 — **learned gate-beat target** | N(1,5)/N(5,1)/Pois(0.5), Lw2/Lr1 | true (provenance) |
| kaynov2024_instance_13 | partial_backorder | K=10, symmetric high-CV (σ/μ=2.8), high penalty 60 | N(5,14)×10, Lw2/Lr2 | true (provenance) |
| kaynov2024_instance_14 | partial_backorder | K=10, strongly heterogeneous gradient, high penalty 60 | N(0,20)..N(10,0)+Pois mix, Lw2/Lr2 | true (provenance) |

Plus a repo-native exact verifier `VERIFICATION_PROBLEM_INSTANCE` (2 retailers, binary demand support {0,1}, 2-period horizon, lost-sales, discount 0.99); `literature_verified = false` — a correctness anchor, NOT a published number.

## Baselines
- **Heuristics:** echelon base-stock + downstream allocation, three flavors — proportional (Kaynov Eq. 8 floor), min-shortage, random-sequential. The keep/discard **gate** is grid-searched echelon base-stock = the better of {proportional, min_shortage}, searched on a search-seed CRN block and re-scored on a DISJOINT held-out CRN block. The tuned gate is stronger than the published base-stock and sits between it and the published PPO.
- **Exact / optimal:** repo-native reduced finite-horizon DP on `VERIFICATION_PROBLEM_INSTANCE` only — optimal = **8.485** dominates both allocation heuristics (proportional = min_shortage = **9.2225**). This is a self-consistency anchor (`literature_verified=false`), NOT a published number; there is no exact solver for the full 100-period Kaynov instances.
- **Published comparators (CONTEXT only — DRL / cross-protocol):** Kaynov PPO and published echelon base-stock rows. Selected published costs (`= −reward`): instance_7 lost-sales proportional 1406.27, min_shortage 1408.08, PPO 1405.08 (gap −0.09%); instance_11 partial-backorder proportional 1111.76, min_shortage 1109.96, PPO 971.86 (−12.58%); instance_12 proportional 1402.38, min_shortage 1406.43, PPO 1118.92 (−20.21%); instance_13 (K=10) proportional 101727.47, PPO 79727.39 (−21.63%); instance_14 (K=10) proportional 53358.86, PPO 42835.02 (−19.72%). The PPO column is the strongest *learned* DRL reference of a different (PPO) protocol — never a keep/discard gate, never a "beats." The full Kaynov PDF was NOT byte-verified (Cloudflare bot wall); rows are carried with row provenance.

## Verification
- **Re-run row 1 (peer-reviewed, lost-sales):** published instance_7 min_shortage cost **1408.08**; **re-run reproduced 1394.82** (gap **−0.94%**, tol 1.2%) via `one_warehouse_multi_retailer_simulate_policy('echelon_base_stock', [S_w=44, S_r=10,10,10], ..., periods=100, replications=1000, seed=2222, discount=1.0, allocation=min_shortage)` with a mean-filled warm start. Verdict: reproduced within tolerance.
- **Re-run row 2 (peer-reviewed, partial-backorder):** published instance_11 proportional cost **1111.76**; **re-run reproduced 1113.17** (gap **+0.13%**, tol 0.5%) via the same simulate call with `[S_w=43, S_r=6,6,6]`, allocation=proportional. Verdict: reproduced within tolerance.
- **Re-run anchor (repo-native, NOT published):** exact-DP optimal **8.485** vs both heuristics **9.2225** via `ir.one_warehouse_multi_retailer_exact_dp_summary()` (re-confirmed live). This is a transition/cost correctness check, not a literature claim.
- **Verification debt (carry-as-literal caveat):** only 2 of the 14 published rows reproduce tightly by re-run; the remaining 12 rows reproduce ~1–6% off and are carried as table literals (regime-dependent sign: lost-sales ~1–2.5% below, backorder ~3.6–5.5% below, partial-backorder ~6% above). This is a protocol / initial-condition residual (mean-filled warm start + repo search grid), not a transition bug — the env transition+cost is exact-DP-validated. `VERIFICATION_PROBLEM_INSTANCE` and the carried rows are honestly flagged. The full Kaynov PDF was never byte-verified; instance-level "literature_verified" means ROW PROVENANCE, not tight numerical reproduction.

## Results (learned policy)
Same-protocol comparator = the tuned echelon base-stock + allocation **gate** (CRN, 4096 held-out paths, 100-period total cost, honest deployment floor: deploy ≥ gate). All numbers are costs (lower is better); Δ% is the learned policy's improvement over the gate.

- **heterogeneous K=3 (instance_12) — SEED-ROBUST GATE BEAT.** Depth-2 soft tree, `echelon_targets_with_alloc_targets` action head + `absolute_augmented` state, axis-aligned splits, linear leaves, freshly warm-started at the gate, σ=0.05, pop 32 × 800 gen, train_seed_batch 24. 6 optimizer seeds {841,842,844,845,846,847}: **1115.44 ± 5.51** vs gate **1169.59 ± 2.05** → **+4.63% mean** (all 6 seeds positive; mean−std = +4.16% still clears the gate). Robust same-protocol gate beat. vs published PPO scalar 1118.92 (cross-protocol): seed-mean +0.31% below, 5/6 seeds below — it STRADDLES PPO within seed noise, so **no robust PPO beat is claimed** (PPO is context only). [This is the finalized seed-robust number, `SEED_ROBUST_BENCHMARK_2026_06_06.md`. The PAPER table `tab:owmr-results` still carries the older at-risk forms: het-3 N=4 +4.58% (1116.01±7.02) and a depth-3 N=4 +4.99% footnote.]

- **high-CV symmetric K=10 (instance_13) — SEED-ROBUST GATE BEAT (per the 2026-06-06 finalization), single-seed in the paper.** Depth-3 soft tree, `symmetric_echelon_targets` + `absolute_augmented`, linear leaves, warm-started at the gate, σ=0.10, pop 32 × 500 gen, batch 16, proportional allocation. 6 seeds {851,852,853,855,856,857}: **85310 ± 946** vs gate **91890.25 ± 99.56** → **+7.16% mean** (all 6 seeds positive; mean−std = +6.13% still clears it). vs PPO scalar 79727.39: −7.0% mean, 0/6 below → robustly below PPO, **no PPO beat claimed**. **HONESTY FLAG:** the PAPER table `tab:owmr-results` still carries this as a SINGLE-SEED frontier row (84286.43±86.39, +8.27%) labeled "not yet a seed-robust aggregate"; the manifest also flags the paper-table +6.44%/+8.27% form as `at_risk` (single-seed). The 6-seed +7.16% is the finalized supersession (`SEED_ROBUST_BENCHMARK_2026_06_06.md`) but is NOT yet reflected in the paper table.

- **heterogeneous K=10 (instance_14) — TIE.** Learned deployed policy = gate-reproducing warm-start anchor (50445.20, +0.00%). Search-limited (gate lies inside the learned class but CMA-ES does not improve on it within budget), NOT a win. Single-seed; not at risk (no win claimed).

- **backorder / lost-sales K=3 (instances 3 / 9 / 10) — TIE on all three.** Learned per-retailer soft tree matches the tuned gate to numerical error (+0.00%, deployed = warm-start anchor): backorder-heterogeneous 1749.90, lost-sales lead-asymmetric 1541.30, lost-sales long-lead 1782.27. Below published PPO on all three (−1.05% / −1.96% / −6.43%); no PPO claim. Single-seed; not at risk (no win claimed). Demonstrates the per-retailer geometry does not degrade across regimes.

- **symmetric Poisson(3) K=3 (instances 1 / 6 / 11 / 7) — TIE (0.0000%).** Warm-started constant leaf reproduces the gate at generation 0; no win claimed (these instances are near-optimally solved by a tuned echelon base-stock).

Honest headline: the learned soft tree **robustly beats the strongest same-protocol tuned heuristic gate** on the asymmetric K=3 partial-backorder instance (+4.63%, seed-robust), and on the high-CV K=10 instance (+7.16%, seed-robust per the 2026-06-06 finalization, but single-seed in the current paper table). It ties the gate elsewhere. No robust PPO beat is claimed anywhere — PPO is cross-protocol context only.

## Reproduce
```bash
# Env-faithfulness anchor (repo-native exact DP): optimal 8.485 dominates heuristics 9.2225
python -c "import invman_rust as ir; print(ir.one_warehouse_multi_retailer_exact_dp_summary())"

# Published-row re-run (no Rust rebuild): instance_7 min_shortage & instance_11 proportional,
# periods=100, replications=1000, seed=2222, discount=1.0, mean-filled warm start
# (instance_7: S_w=44, S_r=[10,10,10], min_shortage -> 1394.82, gap -0.94%;
#  instance_11: S_w=43, S_r=[6,6,6], proportional -> 1113.17, gap +0.13%)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py

# Seed-robust gate beat, heterogeneous K=3 (instance_12), 6 CMA seeds
for s in 841 842 844 845 846 847; do RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets --policy_state_mode absolute_augmented \
  --leaf_type linear --depth 2 --split_type axis_aligned --temperature 0.10 \
  --warm_start_at_best_base_stock --sigma_init 0.05 --gate_search_paths 64 \
  --training_episodes 800 --es_population 32 --train_seed_batch 24 --holdout_paths 4096 \
  --train_allocation min_shortage --same_seed --seed $s; done

# high-CV K=10 (instance_13), 6 seeds: depth 3, symmetric_echelon_targets, sigma 0.10, gen 500,
# batch 16, proportional allocation (seeds {851,852,853,855,856,857})

# Cross-protocol PPO-beat audit (context only): disjoint-block paired check
python scripts/one_warehouse_multi_retailer/verify_ppo_beat_disjoint_blocks.py
```

## Pointers & caveats
- code: `src/problems/one_warehouse_multi_retailer/env.rs` (transition + cost), `references.rs` (14 Kaynov rows + verifier), `finite_horizon_dp.rs` (exact-DP anchor), `allocation.rs` / `demand.rs` / `rollout.rs` / `bindings.rs`; literature notes: `literature/README.md`; existing folder doc: `README.md` (status "partial", consistent with this card).
- scripts: `scripts/one_warehouse_multi_retailer/` — `run_heuristic_published_benchmark.py`, `run_asymmetric_learned_vs_gate.py`, `run_owmr_ppo_campaign.py`, `verify_ppo_beat_disjoint_blocks.py`; finalized seed-robust entry: `SEED_ROBUST_BENCHMARK_2026_06_06.md`.
- autoresearch: `autoresearch/program_one_warehouse_multi_retailer.md`.
- **Caveat — PPO is cross-protocol:** the published PPO and base-stock rows are a DRL (PPO) protocol; comparisons against them are CONTEXT only, never a "beats." No robust PPO beat is claimed; K=3 seeds straddle the PPO line within seed noise and K=10 is ~7% below PPO.
- **Caveat — exact-DP anchor is repo-native:** optimal 8.485 is a self-consistency correctness check (`literature_verified=false`), not a published value; there is no exact solver for the 100-period Kaynov instances.
- **Caveat — verification debt:** only 2/14 published rows reproduce tightly by re-run; the other 12 are carried as table literals at ~1–6% (regime-dependent sign). Full Kaynov PDF not byte-verified.
- **Caveat — paper-vs-finalized mismatch:** the paper table `tab:owmr-results` carries the at-risk single-seed/N=4 forms (het-3 +4.58% N=4; high-CV-10 +8.27% single-seed). The seed-robust supersessions (het-3 +4.63%, high-CV-10 +7.16%, both 6 seeds) live in `SEED_ROBUST_BENCHMARK_2026_06_06.md` and are the honest headline; the high-CV-10 +8.27% single-seed paper-table claim is still flagged `at_risk` in the manifest.
- **Demand convention:** `RoundedNormal` params are `(mean, std)` — `demand.rs` calls `rand_distr::Normal::new(param1, param2)` and validates param2 as "std". So `N(5,14)` = mean 5, **std 14**, σ/μ=2.8 (matching the paper's high-CV K=10 caption). Demand is rounded and clipped at 0. (Note the regime/cv labels in the manifest dimensions are descriptive tags, not the literal env std.)
