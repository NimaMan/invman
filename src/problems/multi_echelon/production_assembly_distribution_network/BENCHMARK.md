# multi_echelon / production_assembly_distribution_network (Pirhooshyaran–Snyder) — benchmark card

**One-line MDP:** state = per-node raw-material + finished-goods inventory, per-relation pipelines, edge and node backorders of a directed production/assembly/distribution network; action = pairwise order-up-to target on each supply relation; transition = ship after lead `L`, **process all raw on arrival** into finished goods, serve downstream demand (shortfall backordered); one-period cost = holding on raw + finished + in-transit + backorder penalty `b·B`; objective = minimize (long-run) average cost.
**Status:** verified_rerun (single-node analytical newsvendor rows) + faithful_unverified (general-network / serial / mixed / assembly topologies — no published cost); snapshot_only_not_rerun DEBT (van Oers 2024 table-only, adjacent). All reference instances literature_verified=false. **Paper:** §sec:pirhoo of learning_inventory_control_policies_es.tex.

## Problem formulation
General production/assembly/distribution supply network (Pirhooshyaran & Snyder 2021). A directed acyclic network of nodes; each node carries raw-material and finished-goods inventory. Each period a **pairwise order-up-to** request is placed on every supply relation (downstream→upstream, after current demand observed). Shipments placed `L` periods earlier arrive; each node **processes all raw material on arrival** into finished goods. External demand (e.g. N(5,1) at the downstream node) is served from finished stock; shortfall is **backordered and carried over**. Holding is charged on raw, finished, AND in-transit stock; backorder penalty `b·B`. State = per-node raw/finished inventory + per-relation pipelines + edge/node backorders. Node modes: `single`, `assembly_and`, `assembly_or`. Objective = minimize average cost over the horizon.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| pirhooshyaran2021_single_node_cases_1-7 | node_mode:single, regime:newsvendor | published analytical OUL 10.67..106.74; published cost 12.71 / 25.42 / 63.56 / 127.11 | false flag, BUT single-node analytical rows reproduced by re-run (~0.005 abs) |
| serial_case3 | 3-node serial chain 0→1→2, demand N(5,1) at node 2 | env's own best pairwise base-stock gate 60.24 (order-up-to [8,7,9]); serial optima table-only | false (faithful-but-no-published-anchor) |
| mixed_scn_fig1_table5 (mixed distribution-and-assembly) | source distributes to two nodes; two assembly nodes serve customer | gate 297.69 (echelon order-up-to [36,13,7]) | false |
| pure_assembly_network | three-layer assembly tree, 4 sources → 2 assembly → 1 final | gate 283.34 (echelon order-up-to [52,52,52]) | false |

## Baselines
- Heuristics: best pairwise base-stock gate (the environment's OWN grid-searched heuristic, NOT a published optimum); single-node newsvendor.
- Exact / optimal: tiny `finite_horizon_dp` on a small serial network (repo-native verifier only).
- Published comparators (CONTEXT): Pirhooshyaran single-node analytical OUL/cost (7 cases, reproduced); serial Clark–Scarf optima are catalog-only (table-only, NOT reproduced by this env). Source paper gives no analytical optimum for mixed/assembly networks (only random-search, DFO, Spearmint, DNN). **Adjacent DEBT (ledger D3): van Oers 2024 Table 1 two-echelon serial-AM rows are a frozen snapshot with no executable env.**

## Verification
- Published numbers: single-node OUL/cost (7 cases) 12.71..127.11. **Re-run reproduced: single-node 7 cases to ~0.005 abs (e.g. case1 10.6745/12.7111 vs 10.67/12.71)** via `python -c "import invman_rust as ir,json; print(json.dumps(ir.production_assembly_distribution_network_literature_benchmark_summary(serial_replications=10000,seed=1234),default=str))"` ; verdict for single-node: **verified_rerun**.
- **DEBT / caveat (faithful_unverified, Group 4):** the general-network / serial protocol reproduces NO published cost. Carried Pirhooshyaran serial/network optima are table-only; the env does NOT reproduce 47.65 / 72.04 under the carried echelon levels (documented local-vs-echelon order-up-to interpretation gap). The serial/mixed/pure-assembly "gates" (60.24 / 297.69 / 283.34) are the environment's OWN best heuristic, NOT published numbers.

## Results (learned policy)
- serial case3: learned beats env's own gate 60.24 — seed123 57.25 (−4.96%), seed321 54.96 (−8.77%), depth-3 57.85 (−3.97%); ≥9× SEM. **single_seed, at_risk=true.** Env-own-heuristic beat (research result, NOT a published-number beat).
- pure-assembly: learned 274.90 vs gate 283.34 → **−2.98%** (~40× SEM). **single_seed, at_risk=true.** Env-own-heuristic beat.
- mixed distribution-assembly: **CORRECTED to gate-match.** 8 CMA seeds → 306.10 ± 22.89 = **+2.82% ABOVE** gate 297.69, 4/8 seeds below. **multi_seed_mean_std, at_risk=false.** The earlier −0.99% was best-of-3; the honest verdict is parity / gate-match, NOT a beat.
- All comparators are the environment's OWN grid-searched best heuristic on a faithful-but-unverified env — research learned-vs-own-heuristic, NOT published-number beats. The serial optimum 47.65 is structurally unreachable by this env's local pairwise policy and is not used as a target.

## Reproduce
```bash
python -c "import invman_rust as ir,json; print(json.dumps(ir.production_assembly_distribution_network_literature_benchmark_summary(serial_replications=10000,seed=1234),default=str))"
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/production_assembly_distribution_network/autoresearch_mixed_distribution_assembly_network.py --budget full --warm_start_flow 10 --seed 7 --run_tag mixed_flow10_verify
# seed-robust mixed audit (the source of the corrected gate-match verdict):
python scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py
python scripts/production_assembly_distribution_network/reproduce_pirhooshyaran_serial_case3.py
```

## Pointers & caveats
- code: src/problems/multi_echelon/production_assembly_distribution_network/{env.rs, finite_horizon_dp.rs, serial_echelon_simulation.rs, demand.rs, rollout.rs, heuristics/, flownet/, literature/, verification.rs, bindings.rs} ; scripts: scripts/production_assembly_distribution_network/ ; autoresearch: policy_search/programs/program_production_assembly_distribution_network.md.
- Only the single-node analytical rows are verified_rerun. The serial/mixed/pure-assembly rows are learned-vs-own-heuristic on a faithful-but-unverified env; mixed is gate-match (the prior −0.99% was best-of-3). van Oers 2024 Table 1 is an adjacent frozen-snapshot debt (D3).
