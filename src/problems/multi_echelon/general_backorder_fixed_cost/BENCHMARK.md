# multi_echelon / general_backorder_fixed_cost (Geevers / Kunnumkal–Topaloglu) — benchmark card

**One-line MDP:** state = per-stock-point on-hand + inbound pipelines + outstanding edge/customer backorders of a directed supplier→warehouse→retailer network; action = vector of per-node order-up-to (base-stock) targets, routed across warehouse–retailer edges; one-period cost = warehouse holding `hʷΣ(Iʷ)⁺` + retailer holding `hʳΣ(Iʳ)⁺` + customer backorder penalty `b·ΣBʳ` (NO fixed/setup cost despite the family name); objective = minimize long-run average cost.
**Status:** verified_rerun (set1 + Kunnumkal–Topaloglu, within published band); snapshot_only_not_rerun DEBT (set2/set3 order-per-edge, +223% NOT reproduced). **Paper:** §sec:genbackorder of learning_inventory_control_policies_es.tex.

## Problem formulation
General-network backorder system (Geevers, van Hezewijk & Mes 2024 "CardBoard Company"; Kunnumkal–Topaloglu divergent) under long-run average cost. `M` uncapacitated suppliers feed `W` warehouses feeding `R` retailers over a fixed directed warehouse–retailer edge set; unit lead times. Retailer `i` faces i.i.d. Poisson(λ) customer demand; unmet demand is **backordered**.

Per-period stages: (i) due shipments arrive at on-hand; (ii) suppliers ship last period's warehouse orders into the unit pipeline; (iii) each warehouse fulfils current retailer orders routed to it, rationing (when short) in increasing order of requesting retailer `(Iʳ − Bʳ)`, the unfilled part becoming edge backorder; (iv) demand realized, unmet part becomes customer backorder; (v) leftover warehouse stock clears edge backorders, leftover retailer stock clears customer backorders; (vi) costs charged on end-of-period stocks/backorders, new orders placed. Action = per-node order-up-to targets `(Sʷ_j, Sʳ_i)`; for set1, retailer raises route to one upstream warehouse edge by historical connection weights. One-period cost (Eq. gbk-cost): `hʷΣ(Iʷ)⁺ + hʳΣ(Iʳ)⁺ + b·ΣBʳ`. **No fixed ordering cost** — the carried family name notwithstanding, the published objective and the env charge holding + backorder only.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| geevers2023_general_set1 (CardBoard) | regime:backorder, topology 4supplier/4warehouse/5retailer | demand Poisson(15); published benchmark 10467; published PPO best 8714 | **true** |
| geevers2023_general_set2 / set3 (order-per-edge) | topology 4w/5r, action order_per_edge | published benchmark 4797; published PPO best 4175 / 3935; NOT reproduced (+223%) | **false** (both set2 and set3) |
| kunnumkal_topaloglu_divergent | regime:backorder, topology 1supplier/1warehouse/3retailer | warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0, no warehouse backorder; demand Poisson(α), α~Uniform[5,15] resampled per period; published benchmark 4059; published DRL 3724 | **true** |

## Baselines
- Heuristics: constant node-base-stock at published levels (the paired gate).
- Exact / optimal: none (heuristic-only family — no exact solver).
- Published comparators (CONTEXT): Geevers set1 benchmark 10467, PPO best 8714 (cross-protocol); set2/set3 benchmark 4797 (NOT reproduced); Kunnumkal–Topaloglu benchmark 4059, DRL best 3724 (cross-protocol).

## Verification
- Published numbers: set1 = 10467; Kunnumkal–Topaloglu = 4059. **Re-run reproduced: set1 = 10384.9 (−0.78%); KT = 3933.3 (−3.1%)** via `python -c "import invman_rust as ir; print(ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('geevers2023_general_set1',replications=200,seed=1234)['mean_cost'])"` (and `...('kunnumkal_topaloglu_divergent',replications=500,seed=1234)`) ; verdict: **verified_rerun (within published band)**.
- **DEBT (snapshot_only_not_rerun, ledger D2):** set2/set3 (published 4797) carried as table-only rows. The order-per-edge / restricted-transition spec exists only in the gated CEJOR full text; the env yields ~15497 = **+223% = NOT reproduced**. literature_verified=false for both. Do not present alongside the verified set1/KT rows without this flag.

## Results (learned policy)
- set1 (CardBoard): learned beats the *reproduced* gate (10354.8 in audit; paper reports 10354.8 paired gate). seed123 → 8034.8±17.6 (−22.4%), seed777 → 7590.7±19.2 (−26.7%); ≫2× SEM, robust to init. **best_of_n, at_risk=true — both seeds beat, but manifest labels this best_of_n; NOT a multi-seed mean±std statement.**
- Kunnumkal–Topaloglu: learned beats reproduced gate (3930.4) by ~37% (seed123 2469.1±7.6 = −37.2%, seed777 2477.9±8.0 = −37.0%). **best_of_n, at_risk=true.**
- The published PPO/DRL figures (8714 / 3724) are **cross-protocol context, NOT head-to-head**: a different learner under the source's own protocol. The learned policy lands below them but under a different protocol — NOT claimed as a PPO/DRL beat.
- The like-for-like claim is the paired same-environment improvement over the reproduced constant node-base-stock gate.

## Reproduce
```bash
python -c "import invman_rust as ir; print(ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('geevers2023_general_set1',replications=200,seed=1234)['mean_cost'])"
python -c "import invman_rust as ir; print(ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock('kunnumkal_topaloglu_divergent',replications=500,seed=1234)['mean_cost'])"
python scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py --reference kunnumkal_topaloglu_divergent --budget full
```

## Pointers & caveats
- code: src/problems/multi_echelon/general_backorder_fixed_cost/{env.rs, heuristics.rs, references.rs, rollout.rs, literature/, tests/, bindings.rs} ; scripts: scripts/general_backorder_fixed_cost/ ; autoresearch: policy_search/programs/program_general_backorder_fixed_cost.md.
- Family name is a misnomer: there is NO fixed ordering cost (holding + backorder only) — the section title reflects the model actually implemented and verified.
- set1/KT are verified_rerun within the published band; set2/set3 are a NOT-reproduced (+223%) snapshot debt (D2). PPO/DRL comparators are cross-protocol context, never "beats." The learned set1/KT improvements are best-of-N (at_risk), not yet a multi-seed mean±std restatement.
