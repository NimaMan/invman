# OWMR asymmetric / high-CV learned-policy results (Kaynov 2024)

Held-out, paired-CRN comparison of a learned soft-tree policy (per-retailer action
geometry) against the strongest in-repo echelon base-stock + allocation **gate**, on
the asymmetric / high-variability partial-backorder instances. Reference learned
comparator: the published Kaynov PPO row. Protocol matches the autoresearch runner
(100-period undiscounted total cost; disjoint search/held-out CRN blocks; allocation
anchors 700000/800000; win only when the paired-difference advantage exceeds its SEM).

Runner: `scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py`
(parallel gate grid + identical training/eval bindings as
`benchmark_learned_vs_heuristic.py`). Outputs in `outputs/` (gitignored) so the
numbers are embedded here.

## Action geometry actually used

- `echelon_targets` (control dim K+1: warehouse target + per-retailer echelon
  base-stock targets) — instance_12, instance_14. Supports both allocations.
- `direct_orders` (control dim K+1: raw per-retailer orders, proportional only) and
  `symmetric_echelon_targets` (gate-matching baseline) — instance_13.
- `vector_quantity` is NOT a binding action mode (it is the soft-tree control mode);
  the env rejects it. So the per-retailer lever is `echelon_targets` / `direct_orders`,
  not `vector_quantity`.

## Result table (held-out, paired CRN; full budget: pop 32 x 600 gen, train_seed_batch 12, 4096 held-out paths)

All costs are 100-period undiscounted total cost (lower is better). Gate = strongest
in-repo grid-searched echelon base-stock + allocation, held-out. Paired diff = gate -
learned on identical CRN paths under the deployed allocation (positive => learned
cheaper); win requires the paired advantage to exceed its SEM. PPO = published Kaynov
Table A.3 cost (= -reward). Gate grid searched on 64 CRN paths (argmin verified stable
vs 96/256); held-out re-score at 4096 paths.

| Instance | Geometry / leaf | Learned (held-out) +/- SEM | Gate +/- SEM | Gap % | Paired diff +/- SEM | Verdict | Published PPO | Learned vs PPO |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| instance_12 | echelon_targets / linear | **1154.09 +/- 2.12** | 1169.59 +/- 2.05 | **+1.33%** | **+15.50 +/- 0.97** | **learned_wins** | 1118.92 | -3.14% |
| instance_12 | echelon_targets / constant | 1168.43 +/- 2.21 | 1169.59 +/- 2.05 | +0.10% | +1.16 +/- 0.41 | learned_wins (marginal) | 1118.92 | -4.42% |
| instance_13 | symmetric_echelon_targets / linear | **85974.79 +/- 88.29** | 91890.25 +/- 99.56 | **+6.44%** | **+5915.47 +/- 49.50** | **learned_wins** | 79727.39 | -7.84% |
| instance_13 | symmetric_echelon_targets / constant | 91890.25 +/- 99.56 | 91890.25 +/- 99.56 | +0.00% | +0.00 +/- 0.00 | tie | 79727.39 | -15.26% |
| instance_13 | direct_orders / constant (no warm) | 138609.69 +/- 222.59 | 91890.25 +/- 99.56 | -50.84% | -46719.43 +/- 234.31 | learned_loses | 79727.39 | -73.85% |
| instance_14 | echelon_targets / linear | 50445.20 +/- 61.90 | 50445.20 +/- 61.90 | +0.00% | +0.00 +/- 0.00 | tie | 42835.02 | -17.77% |
| instance_14 | echelon_targets / constant | 50445.20 +/- 61.90 | 50445.20 +/- 61.90 | +0.00% | +0.00 +/- 0.00 | tie | 42835.02 | -17.77% |

instance_14 gate levels: W=440, per-retailer R=[33,30,28,26,27,30,2,10,29,39]
(proportional). The trained CMA-ES xbest was ABOVE the gate on both leaves (linear
50603.63, constant 51164.33), so the deployed policy fell back to the gate-reproducing
warm-start anchor (= the gate), giving an exact paired tie.

Gate levels: instance_12 W=39, per-retailer R=[5,10,1] (proportional). The published
base-stock rows for instance_12 are 1406.43 (min_shortage) / 1402.38 (proportional) —
the in-repo grid-searched gate (1169.59) is far stronger than Kaynov's reported
base-stock and sits between it and the published PPO (1118.92).

## Verdicts and framing

- **instance_12 — HEADLINE WIN.** The per-retailer `echelon_targets` soft-tree (linear
  leaf), warm-started at the gate's per-retailer base-stock and deployed as the trained
  CMA-ES xbest, beats the tuned in-repo gate by **+1.33%** with a paired advantage of
  **+15.50 +/- 0.97** (~16 SEM) — a robust flip, not a sub-SEM artefact. The constant
  leaf also wins but marginally (+0.10%, ~2.8 SEM). The learned policy exploits
  state-dependent per-retailer deviations from the shared base-stock that the gate
  cannot express. It remains 3.14% above the published PPO, so PPO is still the strongest
  learned policy on this row, but the in-repo learned policy genuinely beats the in-repo
  gate beyond SEM.
- **instance_13 — HEADLINE WIN (state-dependence, not asymmetry).** instance_13 is
  SYMMETRIC (10 identical N(5,14) retailers) but very high CV (sigma/mu=2.8). The
  CONSTANT-leaf symmetric policy (a static shared base-stock) only TIES the gate
  (deployed = warm-start anchor; CMA-ES found no static improvement). But the
  LINEAR-leaf symmetric policy — whose warehouse/retailer order-up-to TARGET depends on
  the inventory state — beats the gate by **+6.44%** (paired **+5915.47 +/- 49.50**,
  ~120 SEM), closing most of the gap to PPO (15.26% -> 7.84%). The exploitable structure
  here is dynamic state-dependence under high demand variance, not per-retailer
  heterogeneity. (The direct_orders per-retailer ablation is the no-warm expressiveness
  control.)
- **direct_orders ablation confirms the warm-start floor is essential.** The raw
  per-retailer order geometry (control dim K+1, no gate-reproducing anchor, proportional
  only) LOSES by -50.84% on instance_13 (138609.69 vs gate 91890.25): without a warm
  start it cannot even find the base-stock region in 600 generations. The usable
  per-retailer lever is the TARGET-based echelon_targets (warm-startable), not raw orders.
- **instance_14 — TIE (search-limited on the hardest instance).** instance_14 is the
  10-retailer STRONGLY ASYMMETRIC instance (clipped-normal demand gradient
  N(0,20)..N(10,0) plus four Poisson retailers). Its per-retailer echelon_targets gate is
  already strong (W=440, R=[33,30,28,26,27,30,2,10,29,39], held-out 50445.20). With the
  generalized per-retailer warm-start anchor, gen-0 reproduces the gate exactly, but
  CMA-ES (pop 32 x 600 gen) found NO improvement on either leaf (trained xbest 50603.63
  linear / 51164.33 constant, both ABOVE the gate), so the deployed policy fell back to
  the anchor -> exact tie. This is SEARCH-limited, not representation-limited: the 10-dim
  asymmetric target space is far harder to improve in budget than instance_12's 3-dim.
  PPO stays 17.77% cheaper. Honest verdict: learned ties the strong in-repo gate; no flip.

## Decisive lever: per-retailer warm-start floor

Without a gate-reproducing gen-0 anchor, the richer per-retailer `echelon_targets`
policy at screening LOST to the gate by -12.79% (instance_12, linear, no warm-start).
Generalizing the warm-start inversion to any control dim (seed the leaves so gen-0 emits
the gate's per-retailer target vector [W, r_1, ..., r_K]) makes the honest floor apply:
the anchor reproduces the gate to 0.0000 on identical CRN paths, and CMA-ES then searches
outward from the gate inside the larger per-retailer class. With the anchor, instance_12
flipped from a -12.79% loss to a +1.33% win.

## Action geometry actually used (and why not vector_quantity)

The binding's `parse_policy_action_mode` accepts only `direct_orders`, `echelon_targets`,
`symmetric_echelon_targets`. **`vector_quantity` is the soft-tree control mode, not a
policy action mode — the binding rejects it** (verified). The per-retailer lever used was
therefore `echelon_targets` (control dim K+1; supports both allocations and warm-start),
with `direct_orders` as the no-warm proportional-only ablation on instance_13.

## Screening validation

- instance_12, echelon_targets, linear, NO warm-start, screening: learned 1316.63 vs
  gate 1167.28 => -12.79% (paired -149.35 +/- 5.08) — learned loses badly without the
  anchor.
- instance_12, echelon_targets, constant, WITH per-retailer warm-start, screening:
  learned 1167.28 = gate 1167.28 (paired +0.00 +/- 0.00) — the anchor floor holds; the
  deployed policy is the gate-reproducing anchor (60 generations insufficient to improve).
  At full budget (600 generations) the trained xbest then beats the anchor and the gate
  (see table).
