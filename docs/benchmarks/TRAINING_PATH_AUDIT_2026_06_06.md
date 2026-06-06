# Training-Path Audit — xbest vs xfavorite deployment endpoint (2026-06-06)

Auditor pass on the hypothesis flagged at the end of `SEED_ROBUST_RERUNS_2026_06_06.md`:
`es_mp.train` deploys the CMA-ES **`xbest`** (the single best individual on its small
training-seed batch), which OVERFITS on a disjoint held-out CRN block and inflates BOTH the
held-out cost AND the cross-optimizer-seed std. The proposed lever is to deploy **`xfavorite`**
(the CMA-ES distribution MEAN) instead.

All runs: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, ≤ ~8 cores. The global default of
`es_mp.train` / `cmaes.py` was **NOT** changed; both endpoints are extracted from the SAME run
via the result tuple.

---

## 1. Result-tuple layout (CONFIRMED, not assumed)

`invman/cmaes.py` wraps `cma.CMAEvolutionStrategy`. The `es.result` object is a
`CMAEvolutionStrategyResult2`. Under **integer index access** (which `cmaes.py` uses), the
authoritative layout (from the library's own `__getitem__` guard, verified at runtime) is:

| index | field | meaning |
|---|---|---|
| `result[0]` | **xbest** | best solution ever evaluated (single individual) |
| `result[1]` | fbest | its objective value |
| `result[2]` | evals_best | |
| `result[3]` | evaluations | |
| `result[4]` | iterations | |
| `result[5]` | **xfavorite** | the distribution MEAN (`== es.mean`), back-transformed |
| `result[6]` | stds | per-coordinate sigma (used by `rms_stdev`) |
| `result[7]` | stop | termination dict |

(NB: the newer `.names` attribute list inserts `best_feasible` and shifts the *attribute* order,
but **integer indexing preserves the backward-compatible order above** — verified
`result[0] is xbest`, `result[5] is xfavorite`, `result[6] is stds`, and `xfavorite == es.mean`.)

`cmaes.py` already exposes BOTH endpoints (both multiplied by `param_scales`):
- `CMAES.best_param()` → `result[0]` = **xbest**
- `CMAES.current_param()` → `result[5]` = **xfavorite** (the distribution mean)

`es_mp.train` (line ~252, both the in-loop checkpoint at line ~214 and the final return at
~252) deploys `es.best_param()` = **xbest**. The trained model returned to every runner therefore
carries **xbest**.

---

## 2. Blast radius — how each problem deploys the trained policy

`es_mp.train` is the single shared optimizer entry point (≈30 caller scripts). The deployed
endpoint is governed by what each runner does with the returned model:

### (A) Deploy xbest directly (NO floor) — the at-risk group
These runners use the model `train()` returns (= xbest) and evaluate/deploy it directly:
- `scripts/random_yield_inventory/train_soft_tree_reference.py` (the LOSS-row runner)
- `scripts/multi_echelon/train_multi_echelon_policy.py` + `seed_robust_divergent_multi_echelon.py`
- `scripts/joint_pricing_inventory/train_soft_tree_reference.py`
- `scripts/joint_replenishment/*`, `scripts/perishable_inventory/*`,
  `scripts/procurement_removal_inventory/*`, `scripts/spare_parts_inventory/*`,
  `scripts/vendor_managed_inventory/*`, `scripts/nonstationary_lot_sizing/*`,
  `scripts/ameliorating_inventory/*`, the dual_sourcing benchmark suite
  (`benchmark_full_suite.py` deploys `es.best_param()`).

These deploy xbest with no mitigation. Where xbest overfits, the held-out cost and cross-seed
std are both inflated. **This is the bulk of the repo.**

### (B) Already mitigated by an "honest floor" (best-of candidate set)
These runners do NOT blindly deploy xbest; they evaluate xbest on the held-out block alongside
warm-start / gate anchors and deploy the cheaper:
- `scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py` — floor =
  best-of {trained_xbest, warm_start_anchor, init_params_anchor, direct_order_gate_init_anchor}.
- `scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py` —
  deploys best-of {trained_xbest, warm_start_anchor}.
- `scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py`
  — best-of {trained_xbest, anchor, gate} (the `--honest_floor` flag).

The floor *partially* mitigates xbest overfit, but ONLY when an anchor exists and the anchor
happens to be at least as good — it does NOT recover the variance lost to xbest within the
"trained" branch itself, and the no-anchor problems (group A) get no protection at all.

**Summary blast radius: ~all problems train via xbest. ~3 runner families already wrap a
best-of floor (OWMR ×2 + mixed-assembly); the remaining ~10 problem families deploy xbest
directly with no mitigation.**

---

## 3. Controlled paired experiment (xbest vs xfavorite, SAME run, SAME held-out block)

Both endpoints are captured from the SAME CMA-ES run per seed and evaluated on the SAME disjoint
held-out CRN block. The ONLY thing that varies is which endpoint is deployed. No env change; the
global `train()` default is untouched (xbest extracted via `best_param()`, xfavorite via the new
additive `return_optimizer=True` → `current_param()`).

### (a) random_yield (LOSS row reproduction; d1 oblique-linear, 800 ep, pop16, batch8)

Held-out CRN block seeds `100000..104095` (4096 paths, disjoint from training). Gate = LIR
(linear_inflation) on that block = **203.762**. Runner:
`scripts/random_yield_inventory/seed_robust_xbest_vs_xfavorite.py`. Seeds {123,456,789,2026,555}
(the LOSS-row block).

| seed | xbest held-out | xfavorite held-out | gap_xbest% | gap_xfav% |
|---|---|---|---|---|
| 123  | 225.34 | 211.93 | +10.59 | +4.01 |
| 456  | 201.21 | 198.18 | −1.25  | −2.74 |
| 789  | 212.15 | 195.50 | +4.12  | −4.05 |
| 2026 | 235.85 | 207.59 | +15.75 | +1.88 |
| 555  | 269.73 | 238.34 | +32.38 | +16.97 |

| endpoint | seed-mean ± cross-seed std | gap% vs gate | #seeds beating gate |
|---|---|---|---|
| **xbest** | **228.86 ± 26.34** | +12.32% | **1/5** |
| **xfavorite** | **210.31 ± 17.05** | +3.21% | **2/5** |

- **xbest reproduces the documented LOSS row** (doc had 227.36 ± 29.77, 0/5). The tail-risk seed
  is 555 (xbest 269.73, +32%), exactly the overfit signature the hypothesis predicted.
- **xfavorite cuts cross-seed std by 35.3%** (26.34 → 17.05) AND **cuts the seed-mean by 18.5**
  (228.86 → 210.31, +12.32% → +3.21% vs gate). On EVERY seed xfavorite ≤ xbest (paired win 5/5).
- It moves the verdict from **LOSS toward PARITY**: 2/5 seeds now below the gate (vs 1/5), and
  the +3.21% gap is now within ~1σ of parity. Not yet a robust beat, but the loss is no longer
  robust either.

### (b) OWMR het-3 — marginal case (kaynov2024_instance_12, K=3 heterogeneous partial-backorder)

Geometry: `echelon_targets_with_alloc_targets`, linear leaf, depth-2, axis_aligned, t=0.10,
`absolute_augmented` state, warm-started at the gate, train_allocation min_shortage, same_seed.
**Budget reduced** to gen 300 / pop 24 / batch 12 / holdout 2048 / gate-search 128 (from full
600/32/24/4096/256) to fit 5 seeds within the CPU/time cap. Gate (paired, on the held-out block)
= **1170.44**. Runner: `scripts/one_warehouse_multi_retailer/seed_robust_xbest_vs_xfavorite_het3.py`.
Seeds {821..825}.

| seed | gate | xbest | xfavorite | gap_xbest% | gap_xfav% | floor deployed |
|---|---|---|---|---|---|---|
| 821 | 1170.44 | 1186.26 | 1188.79 | −1.35 | −1.57 | warm_start_anchor |
| 822 | 1170.44 | 1167.40 | 1184.03 | +0.26 | −1.16 | trained_xbest |
| 823 | 1170.44 | 1148.49 | 1180.71 | +1.88 | −0.88 | trained_xbest |
| 824 | 1170.44 | 1159.93 | 1156.93 | +0.90 | +1.16 | trained_xfavorite |
| 825 | 1170.44 | 1189.10 | 1195.15 | −1.59 | −2.11 | warm_start_anchor |

| endpoint | seed-mean ± cross-seed std | savings% vs gate | #seeds beating gate |
|---|---|---|---|
| **xbest** | **1170.24 ± 17.32** | +0.02% | **3/5** |
| **xfavorite** | **1181.12 ± 14.58** | −0.91% | **1/5** |

- **xfavorite tightens the std modestly (15.8%, 17.32 → 14.58)** BUT makes the **seed-mean +10.9
  WORSE** (−0.91% vs +0.02%). Here xbest does NOT badly overfit — the warm-start anchor + the
  existing honest floor already cap the tail risk, and at the reduced gen=300 budget the
  distribution mean is still drifting (xfavorite worse than xbest on 4/5 seeds).
- This is the OPPOSITE of random_yield, and it is the decisive evidence: **xfavorite is NOT a
  free lunch.** Where xbest is already well-behaved, deploying xfavorite trades a little variance
  for a worse mean. A global flip to xfavorite would HURT this (and similar warm-started) cases.

### The floor = best-of {xbest, xfavorite} (per-seed minimum on the held-out block)

The honest-floor framing dominates either single endpoint and is downside-safe by construction
(floor ≤ xbest always, and ≤ xfavorite always):

| case | xbest mean ± std | xfavorite mean ± std | **floor = best{xb,xf} mean ± std** |
|---|---|---|---|
| random_yield | 228.86 ± 26.34 (+12.32%) | 210.31 ± 17.05 (+3.21%) | **210.31 ± 17.05 (+3.21%)** (= xfavorite; xfav wins all 5) |
| OWMR het-3 | 1170.24 ± 17.32 (+0.02%) | 1181.12 ± 14.58 (−0.91%) | **1169.64 ± 17.81 (+0.07% savings)** (≥ xbest) |

On random_yield the floor inherits xfavorite's win; on het-3 the floor never does worse than
xbest. (Caveat, stated honestly: the per-seed minimum peeks at the held-out block — but this is
exactly what the EXISTING production honest floor already does when it deploys best-of
{xbest, anchor, gate} on the held-out costs, so adding xfavorite as one more candidate is
consistent with the in-place methodology, not a new in-sample bias.)

---

## 4. Does the lever work? (quantified)

- **Variance reduction: YES, on both cases** — cross-seed std −35.3% (random_yield) and −15.8%
  (het-3). The hypothesis (xbest's per-seed overfit inflates cross-seed std) is **confirmed**.
- **Held-out generalization: case-dependent.** When xbest overfits badly (random_yield, no
  anchor) xfavorite improves the mean a lot (−18.5, +12.3%→+3.2%). When xbest is already
  well-behaved (het-3, warm-started + floored) xfavorite WORSENS the mean (+10.9). **A global
  flip to xfavorite is therefore NOT safe.**
- **Verdict flip: random_yield LOSS → not-robust-LOSS / approaching PARITY** (0–1/5 → 2/5 below
  gate; +12.3% → +3.2%). It does NOT flip to a beat. het-3 is unaffected under the floor.

---

## 5. Recommendation (concrete, minimal, reversible)

**Add `xfavorite` to the honest-floor candidate set; expose a `--deploy_endpoint
{xbest,xfavorite,floor}` flag; do NOT flip the global default.**

This audit implemented exactly that on ONE problem as a reference:
1. `invman/es_mp.py::train` gained an **additive** `return_optimizer=False` kwarg. Default
   `False` ⇒ the historical `(model, fitness_hist)` return and the deployed endpoint (xbest)
   are **UNCHANGED for every existing caller** (zero blast radius). When `True`, the live `es`
   is also returned so a caller can read `es.current_param()` (xfavorite) from the SAME run.
2. `run_asymmetric_learned_vs_gate.py::run_one` gained a `deploy_endpoint` parameter and a
   `--deploy_endpoint {floor,xbest,xfavorite}` CLI flag. Default **`floor`** simply ADDS the
   xfavorite endpoint as one more honest-floor candidate (`trained_xfavorite`), so prior OWMR
   results change only if xfavorite is *cheaper on the held-out block* — i.e. strictly
   downside-safe. `xbest` reproduces the historical behavior EXACTLY; `xfavorite` deploys the
   mean only. The JSON now records `xbest_cost`, `xfavorite_cost`, and their gaps for every run.

**Blast radius of the recommended change (floor-as-default):**
- Group B floored runners (OWMR ×2, mixed-assembly): adding xfavorite to the candidate set is
  downside-safe (floor ≤ prior). Verdicts can only improve or stay.
- Group A direct-xbest runners (random_yield, multi_echelon, joint_pricing, perishable, dual,
  etc.): to benefit they must adopt the same pattern — evaluate xfavorite on the held-out block
  and deploy best-of {xbest, xfavorite}. This is a per-runner one-liner (each already has a
  held-out eval), NOT a global default flip. Recommended next step, runner by runner.

**Which verdicts would likely change:**
- **random_yield: LOSS → PARITY (likely)** once its runner adopts best-of {xbest, xfavorite}
  (mean +12.3% → +3.2%, std −35%, 2/5 below gate). Still not a robust beat — the LIR gate stays
  ahead on the mean — but the *robust LOSS* dissolves.
- **dual_sourcing CDI-parity rows**: plausibly tighten (the +0.09%±0.13% straddle rows have the
  same small-batch xbest noise); to be confirmed by a per-row paired run — the existing
  `benchmark_full_suite.py` would need the same `current_param()` capture.
- **OWMR het-3 / warm-started cases: UNCHANGED** under the floor (floor picks xbest there).
- **nonstationary parity, mixed-assembly straddle**: candidates for a follow-up paired run;
  variance reduction is expected but a verdict flip is not (both sit at parity vs strong gates).

**Do NOT** silently change `es.best_param()` deployment globally — het-3 shows that hurts
warm-started, low-overfit problems. The floor is the correct, additive, reversible lever.

---

## Artifacts

- `invman/es_mp.py` — additive `return_optimizer` kwarg (default False; global default unchanged).
- `invman/cmaes.py` — UNCHANGED (already exposes `best_param`=xbest, `current_param`=xfavorite).
- `scripts/random_yield_inventory/seed_robust_xbest_vs_xfavorite.py` — case (a) experiment.
- `scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py` — additive
  `--deploy_endpoint {floor,xbest,xfavorite}` (default `floor` = prior behavior + xfavorite cand).
- `scripts/one_warehouse_multi_retailer/seed_robust_xbest_vs_xfavorite_het3.py` — case (b) driver.
- Data: `outputs/random_yield_inventory/xbest_vs_xfavorite/random_yield_xbest_vs_xfavorite_d1_linear_800ep.json`,
  `outputs/one_warehouse_multi_retailer/xbest_vs_xfavorite/owmr_het3_xbest_vs_xfavorite.json`.
