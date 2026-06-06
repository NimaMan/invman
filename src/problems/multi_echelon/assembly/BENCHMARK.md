# multi_echelon / assembly (Rosling reduction) — benchmark card

**One-line MDP:** state = per-component on-hand + finished-goods on-hand + in-transit pipelines of an assembly system where several components are kitted into one finished item; action = echelon base-stock target per stage; one-period cost = component (kit) holding + finished-goods holding + customer backorder penalty; objective = minimize long-run average cost.
**Status:** faithful_unverified (verified-by-equivalence only via Rosling reduction; every instance literature_verified=false; could NOT re-run via bindings this audit — no assembly env binding). **Paper:** no dedicated section — the literature anchor (Rosling 1989) is structural, not a worked benchmark; the paper's "assembly" appears only inside §sec:pirhoo (the production/assembly/distribution network, a different family).

## Problem formulation
Assembly multi-echelon under long-run average cost. Several upstream component stages each feed a single downstream assembly stage that combines (kits) them into one finished product served to the customer. Equal-lead-time assembly systems admit Rosling's (1989) structural reduction to an equivalent serial system, after which the Clark–Scarf echelon base-stock policy is optimal. State = per-component on-hand + finished on-hand + in-transit pipelines; action = echelon base-stock target per stage; one-period cost = component/kit holding + finished-goods holding + customer backorder penalty `p·B`; objective = minimize long-run average cost.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| two_component_poisson_Lc1 | subfamily:assembly, regime:backorder, components:2 | Poisson(5); kit holding 2, finished holding 3, L_c=L_a=1, p=10; reduces to serial [2,3] L[1,1]; solver-derived cost 22.758925 | **false** (guarded by no_assembly_instance_is_literature_verified) |
| three_component_poisson_Lc2 | subfamily:assembly, regime:backorder, components:3 | Poisson(5); kit holding 3, finished holding 7, L_c=2, L_a=1, p=37.12; shares Snyder&Shen Ex6.1's two downstream stages; solver-derived cost 52.536229 | **false** |
| heterogeneous_components_poisson_Lc2 | subfamily:assembly, regime:backorder, components:2 (heterogeneous) | Poisson(4); component holding [0.5,1.5] (kit holding 2), finished holding 4, L_c=2, L_a=1, p=20; solver-derived cost 27.530177 | **false** |

## Baselines
- Heuristics: Rosling-reduced serial echelon base-stock (reuses the serial exact solver after the assembly→serial remap).
- Exact / optimal: reuses the `multi_echelon/serial` exact Clark–Scarf solver via the Rosling reduction. The resulting per-instance optima (22.759 / 52.536 / 27.530) are **solver-derived, NOT published**.
- Published comparators: none. Rosling (1989) is a STRUCTURAL result (assembly→serial equivalence), not a numeric benchmark; no peer-reviewed paper prints a directly-reproducible cost for these instances.

## Verification
- Published number: none directly reproducible. The verified content is the *structural equivalence* (Rosling 1989), checked in src verification.rs; the instance costs are solver-derived.
- **Re-run reproduced: NOT re-run via bindings this audit.** There is no assembly env / no reduction binding exposed to Python; the audit's manual remap gave 26.55 vs the solver-derived 22.759 (mismatch), so the only positive evidence is the Rust-only in-crate equivalence + env-sim test. Verdict: **faithful_unverified** (Group 4 — faithful trainable env where the *adjacent* serial module is verified, but assembly itself reproduces no published number and was not re-run via bindings).

## Results (learned policy)
- No learned-policy benchmark claim is carried for assembly in the manifest results list. (The verified learned-policy "match" lives in the sibling `multi_echelon/serial` family.) No win or beat is claimed here.

## Reproduce
```bash
# No Python binding for the assembly env / Rosling reduction (audit gap).
# Rust-only equivalence + env-sim checks:
#   cargo test -p invman_rust ...assembly...   (in-crate verification.rs)
python scripts/assembly/verify_assembly_rosling_independent.py
python scripts/assembly/benchmark_assembly_policies.py
```

## Pointers & caveats
- code: src/problems/multi_echelon/assembly/{env.rs, rosling.rs, echelon_base_stock.rs, references.rs, verification.rs} ; scripts: scripts/assembly/ ; autoresearch: none dedicated (covered under autoresearch/program_multi_echelon.md).
- HONEST DEBT: no assembly env binding is exposed to Python, so the costs were NOT reproduced by re-run this audit (manual remap gave 26.55 ≠ 22.759). Treat the 22.759/52.536/27.530 figures as solver-derived self-consistency anchors, not published numbers.
- The honesty guard test `no_assembly_instance_is_literature_verified` enforces that every instance stays `literature_verified=false`; do not promote any of these to a published-number row.
