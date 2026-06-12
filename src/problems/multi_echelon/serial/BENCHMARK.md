# multi_echelon / serial (Clark–Scarf) — benchmark card

**One-line MDP:** state = per-stage on-hand + in-transit pipelines of an `N`-stage serial chain; action = echelon order-up-to level per stage; one-period cost = sum of installation holding `h^i (I^i)^+` over stages plus customer backorder `p·B`; objective = minimize long-run average cost.
**Status:** verified_rerun (TRUE Clark–Scarf optimum; peer-reviewed published anchor Snyder & Shen Ex 6.1 = 47.65). **Paper:** §sec:serial of learning_inventory_control_policies_es.tex.

## Problem formulation
Classical serial multi-echelon (Clark & Scarf 1960) under long-run average cost. `N` stages in series, indexed downstream→upstream: stage 1 faces i.i.d. customer demand; stage `N` replenishes from an ample external source. Deterministic integer lead times on each link.

Timing of a period: (i) shipments placed `L_i` periods ago arrive at each stage; (ii) stage 1 serves `min(I^1, d_t + B_{t-1})`, the unmet part becoming backorder `B_t`; (iii) each stage raises its echelon inventory position toward its echelon order-up-to target, constrained by upstream on-hand; (iv) pipelines advance. State = per-stage on-hand `I^i` and in-transit pipeline. One-period cost = `Σ_i h^i (I^i)^+ + p·B` (installation/local holding, decreasing upstream; single customer backorder penalty). Objective = minimize long-run average holding + backorder cost. The optimum is attained by an echelon base-stock policy and is computable in closed recursive (Clark–Scarf newsvendor decomposition) form.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| serial:snyder_shen_example_6_1 | subfamily:serial, regime:backorder, N:3 | demand Normal(5,1); L=[2,1,1]; echelon holding [2,2,3]; penalty 37.12; published optimum 47.65 | **true** (the one genuinely peer-reviewed published anchor) |
| serial:poisson_N1_N2_N3 | subfamily:serial, regime:backorder, N:1/2/3 | demand Poisson(5); L0=1; reference impl stockpyl.ssm_serial; optima 4.2208/16.7978/72.0435 | reference-implementation-verified (matches stockpyl, NOT paper-printed) |
| serial:two_stage_normal_and_five_stage_normal_poisson | subfamily:serial, N:2/5, demand normal/poisson | optima 166.2705 (Normal(100,15)) / 225.8672 (5-stage Normal(32,5.657)) / 226.8458 (5-stage Poisson(32)); stockpyl problem_6_1/6_2a/6_2b | reference-implementation-verified (stockpyl-derived) |

## Baselines
- Heuristics: exact Clark–Scarf echelon base-stock (this IS optimal here); newsvendor-per-echelon; lead-time-mean.
- Exact / optimal: exact Clark–Scarf recursive-newsvendor decomposition — a TRUE long-run-average optimum that mirrors stockpyl. Ex 6.1 = 47.6654 (re-derived).
- Published comparators: Snyder & Shen Foundations of Inventory Management Example 6.1, optimal 47.65 (proven optimum, like-for-like — the comparator is the optimum, not a DRL/cross-protocol baseline).

## Verification
- Published number: Snyder & Shen Ex 6.1 = 47.65 ; **re-run reproduced: 47.6654** (gap +0.032%) via `python -c "import invman_rust as ir; print(ir.multi_echelon_serial_exact_normal_solution([3,2,2],[1,1,2],37.12,5.0,1.0))"` ; verdict: **verified_rerun**.
- Reference-implementation re-derivations (stockpyl), all re-run this audit: Poisson N1/N2/N3 = 4.2211 / 16.7983 / 72.0467 (vs stockpyl 4.2208/16.7978/72.0435); 5-stage 225.867 and 226.846; 2-stage 166.271. These are reference-implementation-verified (match stockpyl), not paper-table-printed values.

## Results (learned policy)
- Warm-started direct-level (echelon order-up-to) soft tree **reproduces** the proven optimum: 47.6554 vs published 47.65, gap +0.011% (99.99% match), inside the env's ~±0.06% Monte-Carlo reproduction band. **MATCH only — a proven optimum cannot be beaten.** Manifest seed_reporting = `single_seed`, at_risk = false (the claim is a match within the reproduction band, not a win).
- Three further Snyder & Shen instances likewise matched: 166.2705→166.2300, 225.8672→225.8047, 226.8458→226.8447 (all signed gaps ≤0.03% in magnitude). Reported as matches, not improvements.

## Reproduce
```bash
python -c "import invman_rust as ir; print(ir.multi_echelon_serial_exact_normal_solution([3,2,2],[1,1,2],37.12,5.0,1.0))"
python -c "import invman_rust as ir; print(ir.multi_echelon_serial_exact_poisson_solution([3,2,2],[1,1,2],37.12,5.0))"
python scripts/multi_echelon_serial/benchmark_serial_clark_scarf.py
```

## Pointers & caveats
- code: src/problems/multi_echelon/serial/{env.rs, exact.rs, echelon_base_stock.rs, rollout.rs, verification.rs, bindings.rs} ; scripts: scripts/multi_echelon_serial/ ; autoresearch: policy_search/programs/program_multi_echelon_serial.md (also policy_search/programs/program_multi_echelon.md).
- The comparator is a *proven optimum*; the only honest verdict is reproduction, never "beats." Do not read the learned match (47.6554) as an improvement.
- The Poisson and multi-stage rows are reference-implementation-verified against stockpyl, NOT against a printed paper table — keep that distinction.
