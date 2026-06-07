<!--
RESULTS_padn_mixed.md — experiment log for the agentic_policy_search system on the PADN
(production_assembly_distribution_network) MIXED distribution+assembly network. Comparator is the
env's own best pairwise base-stock GATE; PADN has NO published DRL/PPO baseline, so this is a
gate-beat only — never a PPO claim.
-->

# Agentic policy search — PADN mixed distribution+assembly network (2026-06-06)

## Why this needed new machinery
The mixed-network was the lab's open case: hand-search with the `vector_quantity` head only ever
**tied** the gate (seed-mean **306.10 ± 22.89, +2.8%** vs gate, 4/8 seeds below — parity/gate-match,
not a beat). Root cause (audited): `vector_quantity`'s scale-normalized leaf is NOT gate-invertible,
so there is no clean gate-reproducing warm-start and CMA seeds scatter ±22.9. The deferred fix was a
**residual gate-backbone head** (`order = gate_order + Δ`, warm-started at Δ=0).

## What was built (this fast-follow)
- Rust `ResidualBaseStock` head in the shared `soft_tree.rs` (additive; identity-leaf signed residual,
  Δ=0 at zero params → **gen-0 == gate byte-exact**, proven by an in-crate test; full lib suite 176/0).
- The 3 PADN rollout bindings extended with `backbone_levels`/`residual_group_of` (additive).
- A sibling PADN oracle (`evaluate_policy_spec_padn.py`) + DSL compiler (`policy_spec_compiler_padn.py`)
  reusing the PADN gate search + CMA loop, with the OWMR oracle's honest metrics **verbatim** and a
  hard `anchor_cost == gate_cost` assertion (gen-0==gate end-to-end).
- Agent wiring: `padn_system_prompt`, problem-aware niche domains, separate PADN archive, per-problem
  routing.

## Search → result (real Codex)
Screening search (novelty-driven) explored distinct residual niches; the best structure was
`residual_base_stock / linear leaf / oblique / per-relation`. Full-budget confirmation
(`specs/padn_explore_best.json`, 60-gen CMA, fine gate grid, 5 seeds):

| metric | value |
|---|---|
| gate_cost (production, OUL [36,13,7]) | 297.688 |
| mean_cost (deployed, seed-robust) | **291.136** |
| gap vs gate | **−2.20%** |
| per_seed | 287.36, 294.42, 290.74, 293.22, 289.93 |
| seeds below gate | **5 / 5** |
| std_cost | 2.49 (mean+std = 293.63 < gate) |
| **robust_gate_beat** | **TRUE** |
| anchor_cost | 297.688 = gate (deploy floor did not bind) |

**Outcome:** a robust **−2.20% gate-beat** where hand-search only tied — and the **±22.9 seed scatter
collapsed to 2.49**, because the residual head anchors every seed at the gate by construction. This is
the residual-gate-backbone fix realizing exactly the improvement the audit predicted.

## Honest caveats
- **Gate-beat, not a PPO beat.** PADN has no published DRL/PPO baseline; the comparator is the env's
  own pairwise base-stock gate only.
- 5 seeds (mandate met); std is tight (≈0.86% of mean) and all 5 are clearly below the gate, so the
  result is solid — a ≥10-seed firm-up would tighten it further.
- The screening search was a short (cut) chunk; a longer search might find a wider-margin structure.
  Faithful-but-not-literature-verified env (research result on the env's own gate).

## Reproduce
```bash
cd invman && RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python agentic_policy_search/evaluate_policy_spec_padn.py \
  --spec agentic_policy_search/specs/padn_explore_best.json \
  --problem production_assembly_distribution_network --instance 0 --seeds 5 --budget full
# full eval ~6 min; gate fine-grid + 60-gen CMA × 5 seeds. Run under an inner `timeout` (Bash caps at 10 min).
```
