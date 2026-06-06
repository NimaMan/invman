# random_yield_inventory — benchmark card

**One-line MDP:** state = (period, inventory level, pipeline of `L` outstanding orders); action = order quantity placed this period; one-period cost = `c·order + h·(ending)^+ + b·(ending)^-` where the oldest pipeline order arrives in full with prob `p` or not at all (all-or-nothing batch yield); objective = minimize expected discounted total cost over a finite horizon (γ=0.99), full backlogging.

**Status:** `faithful_unverified` / `no_published_number` — self-consistent-only (the reproduced number is the repo's OWN exact-DP anchor, NOT a literature number). **Paper:** none — this system is benchmark-only and is **not** a section of `learning_inventory_control_policies_es.tex` (the paper covers lost sales, dual sourcing, multi-echelon, perishable, general-network backorder, serial Clark–Scarf, OWMR, ameliorating, and PADN).

## Problem formulation

Single-item, periodic-review inventory with **all-or-nothing supply yield** and positive deterministic lead time `L`, finite horizon `T`, discounted cost γ, full backlogging. Structural match to Yan et al. (2026).

Per-period order of events (`env.rs::step_state`):
1. **arrival** — the oldest pipeline order `pipeline[0]` arrives *in full* with probability `p` (success) or *not at all* with probability `1-p`: `realized_arrival = pipeline[0]` if success else `0` (all-or-nothing batch, not per-unit binomial);
2. **demand** — realized: `ending = inventory + realized_arrival - demand`;
3. **cost** — `period_cost = c·round(order)^+ + h·max(ending,0) + b·max(-ending,0)`; reward = `-period_cost`;
4. **shift** — pipeline shifts forward and the new order is appended: `next_pipeline = pipeline[1..] ++ [round(order)^+]`; `period += 1`.

An order placed now arrives after exactly `L` periods. Orders are rounded to nearest non-negative integer. The env itself is **uncapped**; the order cap (max 8) exists only in `finite_horizon_dp.rs` for tractability and is effectively non-binding for the optimal policy (lifting the cap 8→20 moves the optimum only at the 5th significant figure: 40.05990→40.05987). `expected_inventory_position = inventory + p·sum(pipeline)` is the yield-adjusted position used by the LIR/WNH heuristics.

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
| --- | --- | --- | --- |
| VERIFICATION_PROBLEM_INSTANCE (exact-DP slice) | regime:backlog; yield:all_or_nothing; L=2; finite T=5; demand:discrete 6-pt; γ=0.99 | p=0.75, h=1, b=9, max_order_cap=8, init_inv=4, init_pipeline=[3,2] | false |
| PRIMARY_REFERENCE_INSTANCE = `yan2026_style_lt2_p075_discounted` | regime:backlog; yield:all_or_nothing; L=2; finite T=12; demand:Poisson mean 4; γ=0.99 | p=0.75, h=1, b=9, c=1, init_inv=6, init_pipeline=[4,3] | false |
| YAN_2026_SMALL_SCALE_FAMILY (catalog-only) | yield:all_or_nothing; finite discounted; model_match:exact; access:preview_only | numbers_available:false | absent (do_not_use_for_repo_assertions) |
| CHEN_2018_WNH_FAMILY (catalog-only) | yield:all_or_nothing; policy:weighted_newsvendor; access:bibliographic_only | numbers_available:false | absent (do_not_use_for_repo_assertions) |
| INDERFURTH_2015 binomial/proportional grids (DIFFERENT yield model) | yield:binomial_or_proportional; infinite-horizon avg cost; model_match:partial/special_case | numbers_available:true_but_different_model | absent (related_model_aggregate_only) |

## Baselines

- **Heuristics** (both searched via the order-up-to target `S = Poisson((L+1)·mean).invcdf(b/(h+b))`, textbook protection interval `L+1`, with inventory position `X = inv + p·sum(pipeline)`):
  - `linear_inflation` (LIR): faithful rule `q = (1/p)·(S - X)^+`. Exact-DP slice 47.7138 (+19.1% vs optimum, first action 4); PRIMARY sim 203.619 ± 123.769.
  - `weighted_newsvendor` (WNH): yield-weighted expected gap `E_pipeline E_demand[(S - projected)^+]`, **not** inflated by `1/p` (open fidelity question — see Verification). Exact-DP slice 60.3936 (+50.8% vs optimum, first action 8); PRIMARY sim 222.436 ± 66.918 (higher mean, markedly lower variance).
- **Exact / optimal**: exact reduced finite-horizon discounted DP (`finite_horizon_dp.rs::solve_optimal_policy`), full enumeration over demand × yield tree; `lead_time==2` only, discrete demand, capped action. Optimal on VERIFICATION slice = **40.0599**, first action 4.
- **Published comparators (CONTEXT only)**: NONE usable. Yan 2026 and Chen 2018 are paywalled with no public per-instance table; Inderfurth & Kiesmüller 2015 publishes numbers but for a **different yield model** (per-unit binomial / stochastically proportional, infinite-horizon average cost) — only the `F = 1/p` inflation factor carries over, not any cost. All three citations metadata-verified 2026-05 (Crossref/DBLP/RePEc + open working-paper PDF). The Yan 2026 benchmark family also lists `gdsh` and `drl` comparators that the repo does not implement.

## Verification

- **Published number:** none (no public per-instance number exists). **Re-run reproduced:** exact-DP slice optimal_discounted_cost = **40.05989760985441** (first action 4), LIR = **47.71379457283354**, WNH = **60.39357514301890** — matches the README repo-native anchor to full precision; simulation slice LIR = **203.619 ± 123.769**, WNH = **222.436 ± 66.918** over 2000 seeds — matches README exactly. **Re-run via** `invman_rust.random_yield_inventory_exact_dp_summary()` (~0.15s) and `invman_rust.random_yield_inventory_policy_discounted_cost_summary(...)`. **Verdict:** `faithful_unverified` (self-consistent-only).
- **Debt / caveat — self-consistent-only:** the reproduced numbers come from the repo's own exact solver, NOT a literature table. The exact DP was independently re-derived in a from-scratch Python DP of the same MDP and reproduces 40.0598976099 + first action 4, so the code is **implementation-correct**; but there is **no public anchor**, so this is strictly weaker than the verified-by-rerun systems (e.g. lost_sales vs Bijvank 2015 Table 1). This is correctly *not* claimed as `verified_rerun`.
- **Open fidelity question (WNH):** the WNH rule in `heuristics/weighted_newsvendor.rs` computes the yield-weighted gap but does **not** multiply it by `1/p`. Two secondary descriptions (Yan 2026 / Chen 2018 records) state the order is the gap "multiplied by the reciprocal of the mean yield rate". The exact published WNH formula is paywalled and was not recovered, so this was left unchanged (inflating it would push the already-overshooting order further up). Recorded as a precise next step, not a guess.

## Results (learned policy)

- **Carried headline (single-seed, at-risk):** README/experiments report a learned soft-tree (depth 3, linear leaf, 600 ep, CMA-ES population 32) = **196.661** on PRIMARY (2000 held-out seeds), beating LIR (203.619) by 3.4% and WNH (222.436) by 11.6%. This is a **single-seed** result — **NOT yet seed-robust**. The repo's own seed-robust standard (mean ± std over ≥5 seeds) is not met by this headline.
- **Seed-robust slice (4 seeds, still at-risk):** `d1_linear b8` (depth 1, linear leaf, 800 ep, train_seed_batch 8) over 4 seeds {123,456,789,2026} = **196.73 ± 3.35**; all 4 beat LIR (mean **+4.25% ± 1.67%**). This uses 4 seeds, **below the ≥5-seed seed-robust bar**, so it is flagged at-risk.
- **Contradicting saved run (honesty flag):** `tree_primary_d3_linear.json` records soft-tree = **307.01**, which is WORSE than LIR by −48.7%. The headline d=3/600ep number and this saved d=3 artifact disagree; only the `d1 b8 800-ep` configuration robustly beats LIR. Treat the gate-beat as **fragile / config-dependent**, not established.

## Reproduce

```bash
# Exact-DP slice (optimal + LIR + WNH on VERIFICATION_PROBLEM_INSTANCE)
python -c "import invman_rust, json; print(json.dumps({k:v for k,v in invman_rust.random_yield_inventory_exact_dp_summary().items() if k!='verification_reference'}, indent=1))"

# Validate env/heuristics against exact DP
python /home/nima/code/ml/invman/scripts/random_yield_inventory/validate_against_exact_dp.py --simulation_seeds 256

# Full benchmark (exact slice + PRIMARY simulation slice, heuristics)
python /home/nima/code/ml/invman/scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py

# Train one seed of the seed-robust d1/linear/b8 config
python /home/nima/code/ml/invman/scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py \
  --train_soft_tree --depth 1 --leaf_type linear --training_episodes 800 \
  --es_population 16 --train_seed_batch 8 --seed 456

# Aggregate the 4-seed seed-robust mean ± std
python -c "import json,statistics; fs=['/home/nima/code/ml/invman/outputs/random_yield_inventory/tree_primary_d1_linear_b8_s%d.json'%s for s in (123,456,789,2026)]; st=[json.load(open(f))['evaluation']['soft_tree']['mean_cost'] for f in fs]; print('soft_tree %.2f +/- %.2f'%(statistics.mean(st),statistics.pstdev(st)))"
```

## Pointers & caveats

- **code:** `src/problems/random_yield_inventory/` — `env.rs` (MDP), `finite_horizon_dp.rs` (exact reduced DP + heuristic eval), `heuristics/` (LIR `linear_inflation.rs`, WNH `weighted_newsvendor.rs`), `literature/references.rs` (Yan 2026 / Chen 2018 / Inderfurth 2015, all metadata-verified, none anchoring), `rollout.rs` (soft-tree feature map), `verification/tests.rs` (in-crate correctness assertions), `bindings.rs` (Python exports).
- **scripts:** `scripts/random_yield_inventory/` — `benchmark_policies_vs_exact_and_heuristics.py`, `validate_against_exact_dp.py`, `train_soft_tree_reference.py`, `summarize_literature_benchmarks.py`, `common.py`.
- **autoresearch:** none — there is no `autoresearch/program_random_yield_inventory.md` (this system is not part of the autoresearch/paper pipeline).
- **outputs:** `outputs/random_yield_inventory/` holds the saved training artifacts (`tree_primary_d1_linear_b8_s{123,456,789,2026}.json`, `tree_primary_d3_linear.json`, etc.).
- **Honest caveats:** (1) **no published number** — verification is repo-native self-consistency only, status `faithful_unverified`. (2) The learned gate-beat is **single-seed in the README headline** and only 4-seed in the seed-robust slice (below the ≥5-seed bar) — **NOT yet seed-robust**, and one saved d=3 artifact (307.01) is far worse than LIR, so the beat is config-fragile. (3) **WNH `1/p` inflation** is an unresolved fidelity question (paywalled formula). (4) Demand convention: PRIMARY uses Poisson mean 4 (mean, not std); the exact-DP slice uses a discrete 6-point demand. (5) Yan 2026's `gdsh`/`drl` comparators are unimplemented and are CONTEXT only.
