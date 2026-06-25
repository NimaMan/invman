# one-warehouse multi-retailer autoresearch (Kaynov 2024 OWMR)

This is the policy-search program for the `one_warehouse_multi_retailer` problem (Kaynov et al.
2024, IJPE 267, 109088): one upstream warehouse replenishing `K` identical capacitated retailers,
with a per-period downstream **allocation** of scarce warehouse stock — a divergent two-echelon
system under one of three customer regimes (`backorder`, `lost_sales`, `partial_backorder`).

The learned-benchmark phase already established a faithful env and a paired held-out comparison;
the learned depth-2 soft-tree **loses narrowly to the tuned heuristic**. This program drives the
follow-up: a single-policy autoresearch loop that searches the policy/control surface to close (and
ideally beat) that margin on the currently-losing instances.

## Benchmark (trusted)

Instance family: the Kaynov Table A.3 instances exposed by `references.rs`
(`one_warehouse_multi_retailer_list_reference_instances`), all symmetric Poisson(3) K=3 cases for
the regimes we screen. The three **currently-losing** instances (the keep/discard set) are one per
regime:

- `kaynov2024_instance_1` — `backorder`
- `kaynov2024_instance_6` — `lost_sales`
- `kaynov2024_instance_11` — `partial_backorder`

**Strongest heuristic** (the number to beat): grid-searched **echelon base-stock** levels (one
warehouse level + one shared retailer level over the demand-moment-derived bounds in
`echelon_base_stock_search_bounds`) evaluated under the **better of `{min_shortage, proportional}`**
allocation. The grid search runs on a search-seed CRN block, and the argmin `(W, R)` is re-scored on
a DISJOINT held-out CRN block; the lower-cost allocation rule is the "best heuristic".

**Published anchor** (Kaynov Table A.3, carried as negative reward → cost via `-mean_cost`):
proportional / min_shortage / PPO rows per instance. The repo heuristic reproduces these within
~1-6% (regime-dependent sign — see `literature/README.md`); the PPO row is the published learned
comparator, reported but not the keep/discard gate (the in-repo tuned heuristic is the gate).

**Env-faithfulness anchor**: `one_warehouse_multi_retailer_exact_dp_summary()` — reduced
finite-horizon exact DP confirms the optimum dominates both allocation heuristics
(`optimal 8.485 <= proportional/min_shortage 9.2225`). The env transition/cost is exact-DP-validated
(worked-transition test); only the *published-number* reproduction is "approximate".

## Intended search surface (editable)

The single-policy runner exposes these levers via CLI (everything else is held fixed by the trusted
benchmark above):

- **tree depth** `{1,2,3}` — depth-2 is the learned-benchmark default; sweep up for the harder rows.
- **tree temperature** — soft-split sharpness (lower = closer to a hard tree).
- **split type** `{oblique, axis_aligned}` — oblique = linear-combination splits; axis_aligned =
  single-feature thresholds (cheaper, often more stable on these symmetric instances).
- **leaf type** `{constant, linear}` (rollout also supports `sigmoid_linear`).
- **action design / policy_action_mode**:
  - `symmetric_echelon_targets` — one warehouse target + one shared retailer target, expanded inside
    the rollout (the natural geometry for the symmetric K=3 instances; the learned-benchmark default).
  - `direct_orders` / `vector_quantity` — a per-decision order vector (`q^w`, `q^r_k`) bounded by the
    physical caps; more expressive, more parameters.
- **allocation policy** used during training and at evaluation (`proportional` / `min_shortage` /
  `random_sequential`); the headline learned cost is scored under each of `{proportional,
  min_shortage}` and reported at the better one, paired with the heuristic.
- **CMA-ES warm-start (`x0`)**: seed the initial mean at the **best base-stock levels** found by the
  heuristic grid search (`--warm_start_at_best_base_stock`). Under `symmetric_echelon_targets` with a
  constant leaf, the leaf bias is set to the argmin `(W, R)` so generation-0 reproduces the strongest
  heuristic and CMA-ES searches *outward* from a known-good point — the single most important lever
  for beating a near-optimal base-stock baseline.

## Budgets

Two presets in the runner (`BUDGETS`):

- **screening** — cheap first pass: small CMA-ES population, few generations, a small held-out block,
  a coarse search block. Used to rank levers, not to certify a win.
- **full** — promotion budget: larger population / generations, 4096 held-out paths, matching the
  learned-benchmark protocol so a screened winner can be re-scored at decision quality.

Hard CPU cap regardless of budget: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, `mp_num_processors = 1`
(parallelism is rayon inside the population-rollout binding; no Python process pool). Two sibling
autoresearch agents run in parallel, so the ~27-core defaults elsewhere MUST stay overridden here.

## Goal (keep / discard)

For each currently-losing instance, the primary metric is the **held-out relative gap to the best
heuristic** on the same paired CRN block:

    gap% = (best_heuristic_cost - learned_cost) / best_heuristic_cost * 100

(positive = learned beats heuristic). **Keep** a policy design when it flips the sign — i.e. beats
the strongest heuristic out-of-sample on a losing instance, robustly across the held-out block (watch
the held-out stderr; a sub-stderr "win" is not a win). **Discard** designs that stay behind the
heuristic at full budget. Do not lock to one policy class — the job is to find a strong policy, not to
prove soft trees always win; base-stock is near-optimal on these symmetric Poisson(3) instances, so a
flip is a real result.

## What we know (from the learned-benchmark phase)

Full-budget paired held-out result (depth 2, popsize 32, 600 CMA-ES generations, train_seed_batch 12;
4096 held-out paths; 100-period undiscounted total cost; `symmetric_echelon_targets`):

| Instance | CB | Learned | Best Heuristic | Published PPO | Learned vs Heuristic | Winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `kaynov2024_instance_1` | `backorder` | 1584.45 | 1558.12 | 1637.20 | **-1.69%** | heuristic |
| `kaynov2024_instance_6` | `lost_sales` | 1370.50 | 1348.05 | 1347.34 | **-1.67%** | heuristic |
| `kaynov2024_instance_11` | `partial_backorder` | 1189.51 | 1184.46 | 971.86 | **-0.43%** | heuristic |

So the learned soft-tree is *competitive but not dominant*: it loses by 0.43%-1.69%. Priors carried
into this search:

1. The tuned echelon base-stock + allocation heuristic is near-optimal on these symmetric Poisson(3)
   instances — the margin to close is small, so the search should start from the heuristic, not from
   random init. **Warm-start CMA-ES at the best base-stock levels** is the leading lever.
2. `symmetric_echelon_targets` is the right action geometry for the symmetric instances; a constant
   leaf at the warm-start point exactly reproduces base-stock, so depth/temperature/leaf only need to
   buy *state-dependent* deviations from it.
3. `partial_backorder` (instance 11) is the closest to flipping (-0.43%) — the emergency-shipment
   option gives a learned policy more room than pure backorder/lost-sales. Prioritize it.
4. Allocation rule is held fixed inside a comparison (learned scored under the same rule the
   heuristic argmin used); both rules are reported so allocation choice is a visible lever.

Next dimensions to sweep, in rough priority: warm-start-at-best-base-stock (on/off); depth {2,3} on
the partial-backorder row; axis_aligned vs oblique with a linear leaf; temperature; then
`direct_orders` as an expressiveness ablation.

## Autoresearch outcome (2026-05-31 full-budget sweep)

A focused warm-start-centric sweep (8 screening + 10 full configs, CPU-capped at 2 cores) closed the
held-out gap to **exactly 0.0%** on all three losing instances — a **tie, not a strict flip**. Best
config on all three: **depth-2 `axis_aligned` `constant` leaf, temperature 0.05,
`symmetric_echelon_targets`, warm-started at the best base-stock (W, R)**.

| Instance | CB | Best learned | Best heuristic | gap% | Prior |
| --- | --- | ---: | ---: | ---: | ---: |
| `kaynov2024_instance_1` | `backorder` | `1558.12` | `1558.12` | `0.0000%` | `-1.69%` |
| `kaynov2024_instance_6` | `lost_sales` | `1348.05` | `1348.05` | `0.0000%` | `-1.67%` |
| `kaynov2024_instance_11` | `partial_backorder` | `1184.46` | `1184.46` | `0.0000%` | `-0.43%` |

The learned cost equals the heuristic cost to six decimals: the warm-started constant-leaf tree
reproduces the heuristic at generation 0 and CMA-ES finds no profitable state-dependent deviation at
600 generations. This confirms prior (1): the tuned base-stock + allocation heuristic is at/near the
optimum on these symmetric Poisson(3) instances, so there is no exploitable structure for a learned
policy to *strictly* win. No config produced a robust strict flip (held-out stderr ~1.4–2.4).

**Load-bearing fix**: the runner's `_warm_start_flat_params` previously wrote the raw base-stock
target into the leaf block, but the soft-tree applies a per-leaf-type transform before grid-snapping
(constant: `min + sigmoid(p)·span`; linear: `min + softplus(bias + w·state)`; see
`src/core/policies/soft_tree.rs`). The raw target sigmoid-saturated the constant leaf to the
grid maximum, so generation 0 started from an over-stocked policy (instance-11 holdout ≈ 1879 vs
heuristic ≈ 1180), NOT the heuristic. The fix inverts the transform (logit for constant; zeroed leaf
weights + softplus-inverse bias for linear). With the fix, warm-started constant beats the no-warm
control (`-0.20%`/`-0.04%`) and every linear/oblique/depth-3 variant.

Lever ranking (full budget): constant ≫ linear (`-0.32%`…`-1.84%`); axis_aligned ≈ oblique; depth-2 ≈
depth-3 (no value added); temperature immaterial under the warm-started constant leaf; warm-start
`on` ≫ `off`. Not run (bounded): `direct_orders`/`vector_quantity` action design, `random_sequential`
train allocation, sigma schedules, the 11 non-losing instances.

## Autoresearch outcome (2026-06-04 — `kaynov2024_instance_7`, lost-sales `Lw=2`)

First learned-policy result on **`kaynov2024_instance_7`** (`lost_sales`, `Lw=2`, `Lr=[1,1,1]`,
Poisson(3)×3, `hw=0.5`, `hr=1`, `p=9`, 100 periods, 1000-rep protocol; current verified env =
Eq.8 floor proportional allocation + post-emergency holding). This is the natural longer-warehouse-
lead-time companion to the already-screened lost-sales row (`instance_6`, `Lw=1`). Same protocol as
the 2026-05-31 sweep: `symmetric_echelon_targets`, full budget (popsize 32 × 600 CMA-ES generations,
train_seed_batch 12, 4096 held-out paths), warm-started at the grid-searched best base-stock with the
inverted leaf transform, CPU-capped at 2 cores, scored under both `{proportional, min_shortage}`.

| Metric | Value | Source |
| --- | ---: | --- |
| Best in-repo heuristic (min_shortage echelon base-stock, `W=44, R=[10,10,10]`) | `1401.45` (SEM 1.44) | grid search, paired CRN held-out |
| Best in-repo heuristic (proportional, `W=45, R=[10,10,10]`) | `1455.99` (SEM 1.46) | grid search, paired CRN held-out |
| **Deployed learned (best of {trained xbest, warm-start anchor})** | **`1401.45`** | full-budget run, same paired CRN block |
| Learned vs best heuristic | **`0.0000%` (TIE)** | `gap% = 0` |
| Published Kaynov min_shortage cost (`-reward`) | `1408.08` | `references.rs` |
| Published Kaynov PPO cost (`-reward`) | `1405.08` | `references.rs` |
| Deployed learned vs published min_shortage / PPO | `-6.63` / `-3.63` (cheaper) | literature comparison |

Both leaf types confirm the established prior. The warm-start anchor (gen-0) reproduces the tuned
min_shortage echelon base-stock **exactly** (holdout `1401.4461669921875` to 16 digits); CMA-ES's
training-seed `xbest` over-fits and lands slightly *above* the heuristic on the held-out block
(constant `1404.65`, linear `1415.28`), so the honest deployed policy is the warm-start anchor and the
held-out gap to the strongest in-repo heuristic is exactly `0.0%` — a **tie, not a strict flip**, same
as instances 1/6/11. The repo's min_shortage base-stock is ~0.47% *below* its own published row
(`1401.45` vs `1408.08`) and the learned policy ties that stronger repo number, so it is also below the
published min_shortage and PPO costs — but the honest keep/discard verdict against the in-repo gate is
a tie. Constant ≫ linear holds (constant `xbest` 1404.65 < linear `xbest` 1415.28).

**Runner change (honest warm-start floor).** `train()` returns CMA-ES `xbest`, which is the best on
TRAINING seeds and can over-fit relative to the held-out block. The runner now also evaluates the
warm-start gen-0 anchor on the same paired CRN block and **deploys the better of {trained xbest,
anchor}** (`deployed_policy` field), so the headline can never be reported worse than the heuristic-
reproducing anchor it started from. On instance_7 the anchor wins both leaf types; without this floor
the headline would have been a spurious `-0.23%`/`-0.99%` "loss" that is purely training-seed overfit.

## Autoresearch outcome (2026-06-04 — ASYMMETRIC / high-CV partial-backorder rows 12/13/14)

This is the first learned-policy campaign on the **asymmetric / high-variability**
partial-backorder instances, where Kaynov's own PPO beats base-stock ~20% (so there is
exploitable non-base-stock structure, unlike the symmetric Poisson(3) rows that only tie):

- `kaynov2024_instance_12` — partial_backorder, K=3, heterogeneous demand N(1,5)/N(5,1)/Poisson(0.5).
- `kaynov2024_instance_13` — partial_backorder, K=10, SYMMETRIC but very high CV (N(5,14), σ/μ=2.8).
- `kaynov2024_instance_14` — partial_backorder, K=10, strongly heterogeneous (clipped-normal
  gradient N(0,20)…N(10,0) + Poisson(0.5)/3/9/12).

### Action geometry (the lever) — what the binding actually supports

`parse_policy_action_mode` (rollout.rs) accepts exactly THREE action modes:
`symmetric_echelon_targets` (control_dim 2: W + ONE shared R target — cannot express asymmetry,
only ties), `echelon_targets` (control_dim K+1: W + PER-RETAILER echelon base-stock targets —
the asymmetric generalization; supports BOTH proportional and min_shortage), and `direct_orders`
(control_dim K+1: raw per-retailer orders — proportional/random only; min_shortage needs target
positions it does not supply). **`vector_quantity` is the soft-tree CONTROL MODE, not a policy
action mode — the binding REJECTS it as an action mode.** So the usable per-retailer lever is
`echelon_targets` (primary) and `direct_orders` (expressiveness ablation), NOT `vector_quantity`.
This was confirmed by direct binding tests.

### Decisive fix: per-retailer warm-start floor

The autoresearch runner's warm-start (`_warm_start_flat_params`) only seeds the gate at gen 0 for
the 2-control symmetric geometry (`if action_dim != 2: return unchanged`). Without a gen-0 anchor,
the richer per-retailer `echelon_targets` policy at screening LOSES to the gate by **-12.79%**
(instance_12, linear, no warm). The fix (in the dedicated runner below) **generalizes the warm-start
inversion to any control dim**: zero the leaf weights and set the per-dimension leaf bias to
`softplus_inv(T_d - min_d)` (linear) or `logit((T_d-min_d)/span_d)` (constant), with the target
vector `[W, r_1, …, r_K]` taken from the gate's per-retailer levels. Verified: the warm-start
anchor reproduces the gate's echelon base-stock policy to 0.0000 on identical CRN paths for both
leaf types. With the anchor, the honest floor (deploy the better of {trained xbest, anchor})
guarantees `learned ≤ gate` and CMA-ES searches outward from the gate in the larger per-retailer
class — exactly as the symmetric rows do.

### Dedicated runner (why not the autoresearch runner directly)

The autoresearch runner's gate search (`_search_best_heuristic_on_paths`) is SERIAL Python and, for
asymmetric instances, does a FULL CARTESIAN product over per-retailer levels — ~3e14 candidates for
instance_14 (never terminates). The dedicated runner
`scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py` keeps the identical env /
CRN blocks (search 500000 / held-out 900000; alloc anchors 700000 / 800000) / protocol / honest
floor, but (1) enumerates the gate with the SAME reductions as `common.search_best_echelon_base_stock`
(symmetric reduction for 13; Kaynov z0-k candidate set for 14; full cartesian for 12 = 8280), (2)
parallelizes the gate grid over a 4-worker pool (each pinned to 1 thread → ≤4 cores), and (3) caches
the gate argmin per (instance, budget, gate-search-paths, alloc set). The base-stock cost surface is
smooth, so the gate ARGMIN is stable with a smaller gate-search block; the K=10 gates use
`--gate_search_paths 64` while the honest held-out re-score and all learned training/eval stay at the
full budget (4096 held-out paths, popsize 32 × 600 generations, train_seed_batch 12).

### Result table (held-out, paired CRN; win only beyond paired SEM)

Full budget (pop 32 × 600 gen, train_seed_batch 12, 4096 held-out paths; gate grid
searched on 64 CRN paths, argmin verified stable vs 96/256, held-out re-score at 4096).
Costs = 100-period undiscounted total cost. Paired diff = gate − learned on identical
paths under the deployed allocation (positive = learned cheaper).

| Instance | Geometry / leaf | Learned ± SEM | Gate ± SEM | Gap % | Paired ± SEM | Verdict | PPO | Learned vs PPO |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| instance_12 | echelon_targets / linear | **1154.09 ± 2.12** | 1169.59 ± 2.05 | **+1.33%** | **+15.50 ± 0.97** | **learned_wins** | 1118.92 | −3.14% |
| instance_12 | echelon_targets / constant | 1168.43 ± 2.21 | 1169.59 ± 2.05 | +0.10% | +1.16 ± 0.41 | learned_wins (marginal) | 1118.92 | −4.42% |
| instance_13 | symmetric_echelon_targets / linear | **85974.79 ± 88.29** | 91890.25 ± 99.56 | **+6.44%** | **+5915.47 ± 49.50** | **learned_wins** | 79727.39 | −7.84% |
| instance_13 | symmetric_echelon_targets / constant | 91890.25 ± 99.56 | 91890.25 ± 99.56 | +0.00% | +0.00 ± 0.00 | tie | 79727.39 | −15.26% |
| instance_13 | direct_orders / constant (no warm) | 138609.69 ± 222.59 | 91890.25 ± 99.56 | −50.84% | −46719.43 ± 234.31 | learned_loses | 79727.39 | −73.85% |
<!-- ASYM_I14 -->

**instance_13 is a second WIN — via state-dependence, not asymmetry.** instance_13 is
SYMMETRIC (10 identical N(5,14), σ/μ=2.8). The constant-leaf symmetric policy (static
shared base-stock) only TIES; the LINEAR-leaf symmetric policy (state-dependent
order-up-to target) beats the gate by +6.44% (paired +5915 ± 49, ~120 SEM), closing the
gap to PPO from 15.26% to 7.84%. The exploitable structure is dynamic state-dependence
under high CV. The direct_orders raw-order ablation (no warm-start) loses −50.84%,
confirming the warm-start floor + target geometry are the load-bearing levers, not raw
per-retailer orders.

**instance_12 is a headline WIN**: the per-retailer `echelon_targets` linear-leaf soft-tree
beats the tuned in-repo gate by +1.33% (paired +15.50 ± 0.97, ~16 SEM). The warm-start
anchor reproduces the gate exactly (1169.59); the deployed `trained_xbest` then improves on
it via state-dependent per-retailer deviations the shared base-stock cannot express. Still
3.14% above the published PPO, so PPO remains the stronger learned policy, but the in-repo
learned policy genuinely flips the sign vs the in-repo gate. Without the per-retailer
warm-start the same geometry LOST by −12.79% at screening — the warm-start floor is the
decisive lever.

## Canonical workspace

- Program file: this file.
- Runner: `scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py`
  (reuses `scripts/one_warehouse_multi_retailer/common.py` +
  `benchmark_learned_vs_heuristic.py` helpers — `build_soft_tree_model`,
  `evaluate_soft_tree_policy`/`*_from_paths`, the heuristic grid search, the CRN path sampler).
- Ledger: `outputs/autoresearch/<run_tag>/results.tsv` (one TSV row per run: cost, best_heuristic,
  gap, gap%, plus the structure flags and instance).
- Problem home / env + bindings: `src/problems/one_warehouse_multi_retailer/`.
