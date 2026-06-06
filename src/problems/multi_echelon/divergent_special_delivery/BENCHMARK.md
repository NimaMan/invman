# multi_echelon / divergent_special_delivery (Van Roy / Gijsbrechts) — benchmark card

**One-line MDP:** state = warehouse on-hand + warehouse pipeline + per-retailer on-hand + per-retailer pipelines of a one-warehouse, `R`-retailer divergent system; action = `(warehouse order qʷ, shared retailer order-up-to yʳ)` with min-shortage allocation when warehouse stock is short; one-period cost = warehouse holding `hʷ(Iʷ)⁺` + retailer holding `hʳΣ(Iⁱ)⁺` + expedited (special-delivery) cost `cʷ·E` + lost-sales penalty `p·ℓ`; objective = minimize long-run average cost.
**Status:** verified_rerun (constant base-stock anchor, Van Roy rows reproduced within 2%); snapshot_only_not_rerun DEBT on the published A3C relative savings rows. Every reference instance literature_verified=false. **Paper:** §sec:multiechelon of learning_inventory_control_policies_es.tex.

## Problem formulation
Divergent two-echelon system (Van Roy et al. 1997; Gijsbrechts et al. 2022) under long-run average cost. One capacitated warehouse (echelon 0) replenishes `R` identical capacitated retailers, warehouse lead time `l_w`, retailer lead time `l_r`. Retailer demand `d_{i,t} = round(max(0, N(μ,σ²)))` i.i.d.

Per-period stages (faithful Gijsbrechts 2022 convention used for all experiments): (i) arriving warehouse order and retailer shipments merge into on-hand; warehouse places a new order against its **pre-shipment** installation inventory position; (ii) each retailer raises its IP toward shared target `yʳ`; requests filled from warehouse available stock, rationed by a min-shortage (max–min position) allocation `A` when short; (iii) demand realized, unmet `uⁱ`; (iv) each unmet unit independently requests a same-day **special delivery** from leftover warehouse stock w.p. `P_w`, else lost — a hybrid backlog/lost-sales; (v) pipelines advance. State = `(Iʷ; warehouse pipeline; (Iⁱ); per-retailer pipelines)`. One-period cost (Eq. me-cost): `hʷ(Iʷ)⁺ + hʳΣ(Iⁱ)⁺ + cʷ·E + p·ℓ`. The alternative Van Roy 1997 convention (post-shipment installation position, holding on post-decision pre-demand stock) is used ONLY to reproduce the published constant-base-stock costs.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| van_roy1997_simple_problem | divergent, hybrid lost-sales/special-delivery, K:1, mode van_roy_1997 | published const base-stock 51.7; published NDP 52.6 | false |
| van_roy1997_case_study1 (Gijs setting1) | K:10, leadtime l_w2/l_r2 | published const base-stock 1302; published NDP 1179; published A3C savings 8.95% | false |
| van_roy1997_case_study2 (Gijs setting2) | K:10, leadtime l_w5/l_r3 | published const base-stock 1449; published NDP 1318; published A3C savings 12.09% | false |
| gijsbrechts2022_setting1 / setting2 (paper-faithful search targets) | K:10, mode gijs_2022_pre_shipment_eq2, demand_mean 5/0 | no published absolute row; primary_reference_instance = setting2 | false |

## Baselines
- Heuristics: best constant base-stock by grid search; min_shortage (max–min position) allocation.
- Exact / optimal: bounded finite-horizon DP (repo-internal verifier, NOT published) for small instances.
- Published comparators (CONTEXT): Van Roy const base-stock 51.7 / 1302 / 1449; best NDP 52.6 / 1179 / 1318; **Gijs A3C savings 8.95%±0.13% / 12.09%±0.39% (cross-protocol DRL, NOT reproduced — repo implements no A3C)**.

## Verification
- Published numbers: Van Roy const base-stock 51.7 / 1302 / 1449 (under the original Van Roy cost convention). **Re-run reproduced: all three within 2% — 51.77 / 1284.70 / 1437.96** via `python -c "import invman_rust as ir,json; print(json.dumps(ir.multi_echelon_van_roy_reproduction_summary(repo_audit_replications=20,seed=1),default=str))"` ; verdict: **verified_rerun (constant base-stock anchor)**.
- **DEBT (snapshot_only_not_rerun, ledger D1):** the published A3C relative savings rows (8.95% / 12.09%) are carried as snapshot literals and CANNOT be re-run — the repo does not implement A3C. Only the constant base-stock anchor is executable. These rows are published context, NOT reproduced; the drift guards in references.rs assert literals and are NOT verification.

## Results (learned policy)
- setting1: learned 779.81 vs best constant base-stock 911.39 → **−14.44%** (Gijs cost convention). **best_of_n, at_risk=true — single-run/best-of-N, NOT yet seed-robust.**
- setting2: learned 973.55 vs corrected best constant 1137.79 → **−14.43%**. **best_of_n, at_risk=true — NOT yet seed-robust.**
- The cross-method comparison to A3C (8.95% / 12.09%) is INDICATIVE only: different baseline, different cost convention (ours = Gijs 2022; A3C = Van Roy 1997). NOT a like-for-like beat. The grid-action policy (yʷ≤100) stays ~230% above the benchmark — the action-space-trap finding.
- The paper Table tab:me-results carries the −14.4% numbers (779.8 / 973.6); per the manifest these are best-of-N and not yet seed-robust.

## Reproduce
```bash
python -c "import invman_rust as ir,json; print(json.dumps(ir.multi_echelon_van_roy_reproduction_summary(repo_audit_replications=20,seed=1),default=str))"
python -c "import invman_rust as ir,json; print(json.dumps(ir.multi_echelon_gijs_relative_verification_summary(repo_audit_replications=20,seed=1),default=str))"
python scripts/multi_echelon/train_multi_echelon_policy.py --reference gijsbrechts2022_setting1 --budget full
```

## Pointers & caveats
- code: src/problems/multi_echelon/divergent_special_delivery/{env.rs, finite_horizon_dp.rs, exact_rollout.rs, heuristics.rs, references.rs, rollout.rs, bindings.rs} ; scripts: scripts/multi_echelon/ (train_multi_echelon_policy.py, autoresearch_multi_echelon.py) ; autoresearch: autoresearch/program_multi_echelon.md.
- Two cost conventions coexist: Gijs 2022 (pre-shipment merge, holding on end-of-period on-hand — used for all training/experiments) vs Van Roy 1997 (post-shipment installation position — used ONLY to reproduce the published constant base-stock). Do not mix them.
- The A3C savings are cross-protocol DRL context and are NOT reproduced; the learned −14.4% rows are best-of-N and NOT yet seed-robust.
