# multi_echelon — benchmark card index

`multi_echelon` is not a single benchmark but a **family of five structurally distinct multi-echelon
systems**, each with its own faithful environment, baselines, and (different) verification verdict.
Each sub-family has its own self-contained benchmark card; this file only indexes them.
**Paper:** the sub-families map to §sec:multiechelon, §sec:genbackorder, §sec:serial, and §sec:pirhoo of
`paper/learning_inventory_control_policies_es.tex` (assembly has no dedicated section — Rosling 1989 is
structural). **Manifest entry:** the `multi_echelon` block of `docs/benchmarks/BENCHMARK_MANIFEST.json`.
**Ledger:** `docs/benchmarks/VERIFICATION_LEDGER.md` (multi_echelon is intentionally split because the
sub-families have different verdicts).

| Sub-family | Model | Status (per ledger) | Verified anchor (re-run) | Card |
|---|---|---|---|---|
| serial | Clark–Scarf serial chain (Snyder & Shen) | **verified_rerun** (TRUE optimum, peer-reviewed) | Ex 6.1 47.65 → 47.6654 | [serial/BENCHMARK.md](serial/BENCHMARK.md) |
| general_backorder_fixed_cost | Geevers CardBoard + Kunnumkal–Topaloglu network | **verified_rerun** (set1 + KT within band); **DEBT** set2/3 +223% NOT reproduced | set1 10467→10384.9; KT 4059→3933.3 | [general_backorder_fixed_cost/BENCHMARK.md](general_backorder_fixed_cost/BENCHMARK.md) |
| divergent_special_delivery | Van Roy / Gijsbrechts divergent + special delivery | **verified_rerun** (const base-stock ≤2%); **DEBT** A3C savings rows snapshot-only | Van Roy 51.7/1302/1449 → 51.77/1284.70/1437.96 | [divergent_special_delivery/BENCHMARK.md](divergent_special_delivery/BENCHMARK.md) |
| production_assembly_distribution_network | Pirhooshyaran–Snyder general supply network | **verified_rerun** (single-node only) + **faithful_unverified** (general/serial/mixed/assembly) | single-node 7 cases to ~0.005 abs | [production_assembly_distribution_network/BENCHMARK.md](production_assembly_distribution_network/BENCHMARK.md) |
| assembly | Rosling-reducible assembly → serial | **faithful_unverified** (verified-by-equivalence only; all instances literature_verified=false; NOT re-run via bindings) | none reproduced (Rosling 1989 structural; costs solver-derived) | [assembly/BENCHMARK.md](assembly/BENCHMARK.md) |

## Honest verdict summary
- **Genuinely literature-verified-by-rerun against a peer-reviewed published number:** serial (Clark–Scarf
  47.65), and general_backorder_fixed_cost set1 + KT.
- **Verified by re-run within a published band, env-faithful, instances flagged literature_verified=false:**
  divergent_special_delivery constant base-stock rows; padn single-node analytical rows.
- **Faithful but unverified (no published cost reproduced by the trainable env):** padn general-network /
  serial / mixed / pure-assembly topologies; the assembly sub-family.
- **Standing verification debts (snapshot_only_not_rerun):** divergent A3C savings (D1), gbk set2/3 +223% (D2),
  and the adjacent van Oers 2024 padn rows (D3) — asserted literals, NOT executed.

## Learned-policy results — honesty at a glance
- Reproductions/matches (not wins): serial matches the proven optimum to +0.011% (single_seed, NOT at risk).
- Best-of-N / single-seed (at_risk, NOT yet seed-robust): divergent −14.4% both settings; gbk set1 −22.4%/−26.7%;
  gbk KT ~−37%; padn serial case3 and padn pure-assembly env-own-heuristic beats.
- Seed-robust correction: padn mixed distribution-assembly is **gate-match** (8 seeds 306.10±22.89, +2.82% above gate,
  4/8 below) — the earlier −0.99% was best-of-3.
- All PPO/A3C/DRL comparators are **cross-protocol context, never "beats."**

## Pointers
- code: src/problems/multi_echelon/{serial, assembly, divergent_special_delivery, production_assembly_distribution_network, general_backorder_fixed_cost}/ ; top-level mod.rs + bindings.rs.
- scripts: scripts/multi_echelon/, scripts/multi_echelon_serial/, scripts/assembly/, scripts/production_assembly_distribution_network/, scripts/general_backorder_fixed_cost/.
- autoresearch: policy_search/programs/program_multi_echelon.md, program_multi_echelon_serial.md, program_production_assembly_distribution_network.md, program_general_backorder_fixed_cost.md.
