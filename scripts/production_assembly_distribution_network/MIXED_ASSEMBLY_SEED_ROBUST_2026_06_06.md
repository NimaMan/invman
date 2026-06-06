# Mixed distribution-and-assembly network: seed-ROBUST learned-vs-gate (2026-06-06)

> Replaces the paper's best-of-3 seed cherry-pick for the mixed distribution-and-assembly
> network row with an HONEST seed-averaged comparison over the **paper's actual config**.
> Env: `production_assembly_distribution_network`, instance
> `pirhooshyaran2021_mixed_scn_fig1_table5` (faithful, **not** literature-verified).
> All numbers are held-out 4000-path paired-CRN per-period cost (the autoresearch runner's
> protocol), so the gate 297.69 is directly comparable.

## CORRECTION NOTE (supersedes the earlier flow=5 draft)

An earlier draft of this doc took the runner's *default* `--warm_start_flow 5` as "the paper's
exact config" and reported a baseline seed-mean of 380.6 ± 31.1 (+27.8%, 0/5 below gate). **That
was the wrong warm-start.** The committed ledger
(`outputs/autoresearch/mixed_distribution_assembly_network_autoresearch/results.tsv`) shows the
paper's 294.73 (−0.99%) came from `warmstart_flow10.0` — a flat flow ≈ 2× the demand mean (5),
which is the throughput a distribution-split + assembly-`min()` network actually needs. The
honest analysis below uses **flow=10**, the paper's real config. (flow=5 starves the network:
gen-0 ≈ 864/period; it is simply a bad warm-start, not the paper baseline.)

## TL;DR verdict

- **The paper's headline (−0.99%, 294.73, "best of three CMA seeds") is NOT seed-robust.** At
  the paper's flow=10 config, **8 independent CMA seeds** give learned seed-mean
  **306.10 ± 22.89 → +2.82% ABOVE the gate (297.69)**, with **4/8 seeds below the gate**. The
  learned tree **straddles** the gate with large optimizer-seed variance; the −0.99% was the best
  of the paper's three flow=10 seeds (which were 294.73 / 295.30 / 306.25).
- **It is therefore NOT a robust beat** — the seed-mean is above the gate — but it is also **not
  "cannot beat"**: several seeds dip well below the gate (best 277.70 = −6.71%). The honest
  characterization is **parity / within seed noise**, and with an honest deployment floor (deploy
  the better of {trained, gate}) the deployed policy is ≈ the gate (a gate-match).
- **Recommended benchmark entry: ship the gate as the policy for this topology; report the
  learned soft tree as a non-robust straddle (seed-mean +2.8%, 4/8 below), NOT a −0.99% beat.**

## Why it is seed-fragile (root cause — ties to the residual-policy idea)

The soft tree emits a `vector_quantity` (raw per-relation order). For a linear leaf the env
computes `order = min + softplus(bias + w·(state/scale))`, where `scale` is a **dynamic** per-step
normalizer (`build_policy_state`, env.rs). Because every feature is divided by a changing scale,
the leaf **cannot exactly reproduce the gate's affine `order = clip(level − inventory_position)`**.
Consequence: there is **no clean gate-reproducing warm-start** here (unlike OWMR's
`echelon_targets` head, whose explicit target position is exactly invertible —
`run_asymmetric_learned_vs_gate.py::_warm_start_flat_params`). So every seed starts from a flow
anchor that is *near* but not *at* the gate, and CMA-ES wanders — the seeds scatter ±22.9 around
the gate (4 below, 4 above). The fragility is an **anchoring** problem, not a hard representational
wall (the geometry *can* express sub-gate policies — 4/8 seeds find one).

This is exactly the case for a **residual / gate-backbone policy head** (`action = base_stock_gate(state)
+ Δ_tree(state)`): an explicit backbone would anchor every seed *at* the gate and turn this
straddle into a robust ≥-gate result. That is a policy-architecture change (a new action head),
deferred as a follow-up; it is the principled fix for this row.

## Honest seed-robust result — paper's flow=10 config, 8 seeds

Original runner `autoresearch_mixed_distribution_assembly_network.py --budget full
--warm_start_flow 10` (popsize 24, gen 60, train_batch 96, held-out 4000), depth-2 oblique
linear, temp 0.25, σ=0.8. Gate = **297.69 ± 0.67**, OUL [36,13,13,7,7,7,7,36] (echelon
[36,13,7]) — deterministic.

| seed | learned cost | vs gate |
|---|---|---|
| 7   | 277.70 | **−6.71%** |
| 555 | 285.07 | **−4.24%** |
| 777 | 294.73 | −0.99% |
| 321 | 295.30 | −0.80% |
| 123 | 306.25 | +2.88% |
| 999 | 313.07 | +5.17% |
| 42  | 333.48 | +12.02% |
| 888 | 343.18 | +15.28% |
| **seed-mean ± std** | **306.10 ± 22.89** | **+2.82%** |
| frac below gate | **4/8** | — |

(Seeds 123/321/777 are the paper's three flow=10 runs from the committed ledger; 7/42/555/888/999
were added this session via run_tag `mixed_flow10_verify`.) Honest-floor deployed mean
(min(trained, gate) per seed) ≈ 292.9 ≈ the gate — i.e. a gate-match.

### Warm-start value dominates the outcome (the warm-start question, concretely)
- flow=5 (runner default): seed-mean ≈ 380 (+28%) — starves the network (gen-0 ≈ 864/period).
- flow=10 (paper config, ≈2×demand): seed-mean 306.1 (+2.8%), straddles the gate.
- gate-OUL as a *constant* order: ≈ 345 (+16%) — over-orders (≈444/period).
The "result" is driven more by the warm-start anchor than by the learning — the central reason to
make the gate a *structural* backbone (residual head) rather than an optimizer initialization.

## Recommended paper wording (replace the best-of-3 row)

`tab:pirhoo-results` mixed-network row — replace the `−0.99% (best of 3)` line with:

> Mixed distribution-and-assembly network (gate 297.69, echelon order-up-to [36,13,7]). Learned
> soft tree, depth 2, **seed-averaged over 8 independent CMA seeds: 306.1 ± 22.9 (+2.8% vs gate,
> 4/8 seeds below)**. Unlike the serial and pure-assembly networks, on the mixed network the
> learned `vector_quantity` soft tree does **not robustly beat** the environment's own best
> pairwise base-stock — it straddles the gate with large optimizer-seed variance (the earlier
> −0.99% was the best of three seeds). We report the seed-mean, not best-of-N, and treat this row
> as a **gate-match**.

Prose (~lines 3593-3595 of `learning_inventory_control_policies_es.tex`): replace "beating the
gate by up to −0.99% … on the best of three CMA seeds while a third seed sits +2.9% above it"
with: on the mixed network the learned tree is at **parity** — over 8 independent seeds its
seed-mean is +2.8% above the gate (4/8 seeds below), so we report a gate-match, not a beat (the
−0.99% was a best-of-three artifact). Do **not** touch the serial (−4.96%/−8.77%) or pure-assembly
(−2.98%) rows here — those are separately audited (R2: also re-run as multi-seed means).

## Reproduce

```bash
# paper's flow=10 config, 8 seeds (123/321/777 are in the committed ledger; rest added this session)
for s in 7 42 555 888 999; do RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/autoresearch_mixed_distribution_assembly_network.py \
      --budget full --warm_start_flow 10 --seed $s --run_tag mixed_flow10_verify \
      --description "flow10 seed-robust verify seed $s"; done
```

JSON/ledger: `outputs/autoresearch/mixed_flow10_verify/results.tsv` +
`outputs/autoresearch/mixed_distribution_assembly_network_autoresearch/results.tsv` (gitignored;
numbers transcribed above).

## Status of the earlier flow=5 design sweep (seed_robust runner)
The `seed_robust_mixed_distribution_assembly_network.py` sweep (designs A–E, seeds {11,22,33,44,55})
used a gate-OUL-constant or flat-flow=5 warm-start — **both off-target** vs the flow=10 paper
config — so its "+16% best trained seed-mean, 0/5 below gate" headline is superseded by the flow=10
table above. The runner + its honest-floor machinery remain useful (and the honest floor is the
right deployment device), but the canonical comparison is flow=10.
