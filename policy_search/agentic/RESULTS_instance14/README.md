<!--
RESULTS_instance14/README.md — experiment log for the policy_search/agentic system on OWMR Kaynov
instance_14 (K=10 strongly heterogeneous, partial backorder). Records the Codex-driven
policy-structure search and its honest, seed-robust outcome. Comparator is the in-repo
echelon-base-stock GATE; the published PPO scalar is cross-protocol CONTEXT only, never a beat.
-->

# Agentic policy search — OWMR `instance_14` results (2026-06-06)

**System:** `beden`+Codex agent proposes policy-structure specs (README DSL); each is compiled to an
invman soft-tree policy, warm-started at the gate-invertible anchor, trained by an inner CMA-ES on
the Rust rollout oracle, and evaluated seed-robustly vs the gate. One Codex call + one evaluation
per generation. Honest metric: `robust_gate_beat` = (all ≥5 seeds < gate) AND (mean+std < gate);
`deployed_cost` = min(trained, gate). PPO is cross-protocol context only.

## Why instance_14
The spearhead from the seed-robust audit: the one instance where the prior **hand-search only TIED**
the gate (deployed = gate-reproducing warm-start anchor, because trained xbest > gate), yet
exploitable structure was proven on siblings 12/13. Goal: have the agent discover a structure that
robustly beats the gate here.

## Search trajectory (small / screening budget; 2 runs, 10 generations)
`gate_cost` (small budget) = 50702.39.

| run-gen | head | split | leaf | per-ret | mean | 5-seed below | gap% | robust |
|---|---|---|---|---|---|---|---|---|
| 1-0 anchor | echelon_targets | axis | linear | yes | 50641.67 | 5/5 | −0.120 | False |
| **1-1** | echelon_targets | **oblique** | linear | yes | **50632.36** | 5/5 | −0.138 | **TRUE** |
| 1-2 | +alloc_targets | oblique | linear | yes | 50650.25 | 5/5 | −0.103 | TRUE |
| 1-3 | echelon_targets | oblique | linear | yes | 50652.18 | 4/5 | −0.099 | False |
| 2-1…2-4 | echelon_targets | oblique | linear | yes | ~50697 | 4/5 | −0.042 | False (plateau) |
| **2-5** | echelon_targets | oblique | linear | yes | **50632.36** | 5/5 | −0.138 | **TRUE** |

- **3/10 robust gate-beats.** Best structure found = **oblique-split, per-retailer, linear-leaf
  `echelon_targets`** (`specs/winner_gen1_oblique_perret_linear.json`), discovered by Codex's own
  mutation of the axis-aligned anchor → oblique splits.
- **Reproducible**: the exact winner (50632.3594) appeared independently in run-1 gen-1 and run-2
  gen-5 → a stable small-budget optimum, not a lucky seed.
- **Plateau / loop limitation**: Codex re-proposes the same structure once it is the archived best —
  the naive loop has no novelty/temperature/anti-repeat pressure. Clear next improvement to the harness.

## Full-budget confirmation of the winner (the reportable result)
`--budget full`: `gs64` gate, 600-gen inner CMA-ES, **4096 held-out paths**, 5 optimizer seeds.
`gate_cost` (full budget) = **50445.20** (matches the prior campaign's production gate; the
warm-start anchor reproduces it exactly).

| metric | value |
|---|---|
| mean_cost (deployed, seed-robust) | **48168.41** |
| gap vs gate | **−4.51%** |
| per_seed | 46993.67, 47098.19, 49980.18, 50076.74, 46693.29 |
| seeds below gate | **5 / 5** |
| std_cost | 1524.84 |
| mean+std < gate | 49693 < 50445 ✓ |
| **robust_gate_beat** | **TRUE** |
| mean_trained_cost | 48168.41 (deploy floor did NOT bind — training genuinely beat the gate) |

**The screening win held and amplified at production budget: a robust gate-beat of −4.51% on
instance_14**, the previously-unsolved spearhead. The extra training (600 vs 40 generations) let the
oblique/per-retailer structure realize a much larger gain than screening showed.

## Honest caveats (no overclaim)
- **High optimizer-seed variance** (std ≈ 3.2% of mean). The 5 seeds are bimodal: three land at
  ≈ −6.6% to −7.4%, two only ≈ −0.7% to −0.9% below gate. All 5 beat the gate and mean+std < gate, so
  `robust_gate_beat` holds — but the realized policy quality is seed-sensitive. **Next step: re-run at
  ≥10 seeds** to firm up the −4.51% mean before treating it as the final number.
- This is a **gate-beat, not a PPO beat.** Published PPO (Kaynov 2024) ≈ 42835 is still ~11% below our
  48168; it is cross-protocol context (single scalar, never re-trained here, sign-flip + unverified
  N(μ,σ) convention), so no head-to-head PPO claim is made.
- Honest deployable number is the seed-robust **mean (48168.41)**, not the best seed (≈46693, which
  would be best-of-N and against the lab standard).

## UPDATE — novelty-driven longer search found a markedly better structure (2026-06-06)

After adding novelty/diversity pressure (MAP-Elites diverse-elite context + tried-signature anti-repeat
prompt), a longer 10-generation run covered **10/12 valid structural niches** (0 wasted
`symmetric_echelon_targets` proposals — the constraint fix worked) and surfaced a structure the first,
short, fixated run never tried: **constant-leaf** (not linear). It also confirmed `direct_orders` is
unusable (+224%/+671% — no gate-invertible anchor).

Full-budget confirmation of the explored best (`specs/explore_best_constant_oblique.json`):
**`echelon_targets · per-retailer · CONSTANT leaf · oblique · depth-2`**

| structure | full-budget mean | gap vs gate | seeds below | std | robust |
|---|---|---|---|---|---|
| linear/oblique (first winner) | 48168.41 | −4.51% | 5/5 | 1524.8 | TRUE |
| **constant/oblique (explored)** | **44170.26** | **−12.44%** | **5/5** | **450.4** | **TRUE** |

The novelty-discovered structure has **~2.8× the margin AND ~3.4× tighter variance** (per_seed
43686–45025, all far below gate 50445.20; `mean+std = 44621 ≪ gate`; deploy floor did not bind). So
longer exploration with novelty pressure clearly beat the structure the short run had settled on — the
core value of the novelty mechanism, demonstrated.

Honest framing: still a **gate-beat, not a PPO beat** — but it closes the gap to the published PPO
scalar (Kaynov ≈ 42835) from ~12% (linear winner) to **~3%** above; PPO remains cross-protocol context,
no head-to-head claim. 5 seeds (mandate met); the tight std makes this far more solid than the linear
winner, and a ≥10-seed re-run would firm it further. **New incumbent best = constant/oblique.**

**Firmed at 10 seeds (2026-06-06):** re-ran the constant/oblique winner at `--budget full --seeds 10`
→ mean **44105.01**, gap **−12.57%**, **10/10** seeds below gate, std **337.3** (≈0.76% of mean —
*tighter* than the 5-seed std of 450). per_seed all 43686–45025, every seed far below the gate. The
result holds and tightens: a solidly seed-robust **−12.6% gate-beat** on the previously-unsolved
`instance_14`. (Closes to ~3% above the cross-protocol PPO scalar; still not a head-to-head PPO claim.)

## Reproduce
```bash
# winner spec: policy_search/agentic/specs/winner_gen1_oblique_perret_linear.json
cd invman && RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 python policy_search/agentic/evaluate_policy_spec.py \
  --spec policy_search/agentic/specs/winner_gen1_oblique_perret_linear.json \
  --problem one_warehouse_multi_retailer --instance 14 --seeds 10 --budget full
# full budget takes ~10 min (gs64 gate search dominates) — run under Monitor, not a 10-min Bash.
```
