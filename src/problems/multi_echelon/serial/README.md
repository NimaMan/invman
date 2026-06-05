# multi_echelon / serial — textbook serial multi-echelon (Clark & Scarf)

Canonical, literature-faithful home for the **textbook serial multi-echelon inventory
system** (Clark & Scarf 1960). It is the `serial` *version* of the multi-echelon problem;
siblings under `multi_echelon/` cover other topologies. This is the clean model — named
for exactly what it is — that we train policies on.

It is a distinct sibling of **`multi_echelon/production_assembly_distribution_network`**, which implements the richer
Pirhooshyaran & Snyder (2021) general supply-network model (per-node production steps and
pipeline holding) and does **not** reduce to this textbook serial system.

## Problem

- `N` stages in series, indexed downstream → upstream. Stage 1 (downstream) faces i.i.d.
  customer demand; stage `N` (upstream) replenishes from an outside source with ample stock.
- Deterministic integer lead times on each link; linear installation (local) holding cost per
  stage; backorder penalty at the customer.
- Optimal policy: **echelon base-stock** (Clark & Scarf 1960).
- Objective: minimize long-run average holding + backorder cost.

## Package layout

- `env.rs` — the clean serial environment used for policy training. Period sequence is
  **receive → demand → cost → replenish**; orders are placed *after* demand is observed (the
  L-period lead-time-demand convention; ordering before demand is the classic off-by-one error).
  Holding is charged on physical on-hand only (in-transit pipeline is not charged, matching the
  optimized Clark-Scarf cost). Exposes `consume` / `replenish` (two-phase, for observe→act
  training) and a raw state vector.
- `exact.rs` — exact Clark-Scarf recursive newsvendor decomposition: optimal echelon base-stock
  levels and optimal cost. Mirrors Snyder's `stockpyl.ssm_serial`.
- `echelon_base_stock.rs` — the optimal echelon base-stock policy and a Monte-Carlo evaluator.
- `verification.rs` — the confidence checks (below).
- `scripts/multi_echelon_serial/benchmark_serial_clark_scarf.py` — runnable benchmark
  (faithful Python port of `env.rs` + `exact.rs`) comparing the optimal echelon base-stock
  policy against base-stock heuristics on the verified instance set; see *Benchmark* below.

## Verification (env reproduces the literature)

Status: **PARTIAL** — be precise about which block is verified against what:

- **Exact solver vs a genuinely published number — literature-verified.** `exact.rs` re-derives
  the Snyder & Shen *Fundamentals of Supply Chain Theory* **Example 6.1** optimum, the only
  textbook-PUBLISHED anchor here (3-node, echelon holding `[2,2,3]`, lead times `[2,1,1]`,
  stockout 37.12, Normal(5,1)): published cost **47.65**, solver `47.6654`. Independently
  cross-checked against the textbook author's own reference implementation `stockpyl.ssm_serial`
  (`example_6_1`), which reports `C* = 47.6687`, `S* = {6.514, 12.012, 22.700}` — agreement to
  ~0.006%. So a published paper/textbook value IS re-derived by a solver (not merely stored).
- **Exact solver vs the reference implementation (Poisson) — reference-implementation-verified,
  NOT a published-paper anchor.** The Poisson 1/2/3-stage optima (`C* = 4.220849` / `16.797779`
  / `72.043543`, `S* = 8` / `[7,13]` / `[9,15,26]`) are repo-CONSTRUCTED instances, not numbers
  printed in any paper; they match `stockpyl.ssm_serial.optimize_base_stock_levels` to machine
  precision. This is verification against a public reference implementation (strong), but it is
  not a published benchmark number — do not call it "literature-verified" without that qualifier.
  See `exact.rs` tests `single_stage_reduces_to_newsvendor_closed_form` and
  `poisson_instances_match_reference_implementation`.
- **Env-simulation vs the exact solver — verified only for downstream lead time = 1, with a
  documented Normal-demand bias.** `env.rs` driven by the optimal echelon base-stock policy
  reproduces the exact Poisson optima by Monte-Carlo within sampling error (`4.211`, `16.777`,
  `72.007`, all ≤0.23%). The Example 6.1 **Normal** simulation does NOT cleanly reproduce 47.65:
  the evaluator rounds Normal demand and the simulated cost is **≈48.44 (+1.62%)** — the
  `verification.rs` Ex6.1 test passes only because its tolerance is 2% (see Caveat 1; with the
  rounding removed it is +0.01%). And for any instance whose demand-facing stage has lead time
  ≥ 2 the simulation under-counts cost (see Caveat 2). `exact_and_simulation_agree` cross-checks
  decomposition vs simulation directly. Net: the env is a faithful Clark-Scarf training env on
  the carried L₀=1 instance set, but its simulation is a self-consistency check against the
  in-repo exact solver, not an independent reproduction of a published number.

This is the pre-training correctness gate: before any learned policy is trained on `env.rs`, the
env is shown to reproduce the literature optimum under the known-optimal policy.

### Diversifying instances (2026-06-05): more stages, Normal + Poisson

To broaden the learned-policy benchmark beyond the single 3-stage Example 6.1, three additional
serial instances from Snyder & Shen / `stockpyl` are carried (all downstream `L_1 = 1`, the faithful
regime). They are **reference-implementation-verified** (matched to `stockpyl.ssm_serial`), NOT
separately published-paper anchors, so the comparator is the in-repo exact solver value (still a
proven Clark-Scarf optimum -> match-only). Each is asserted in `verification.rs` (exact solver value
+ env-sim reproduction under the exact echelon levels):

| instance        | demand        | install. holding (d->u) | echelon holding (d->u) | L           | p  | exact C\*  |
|-----------------|---------------|-------------------------|------------------------|-------------|----|------------|
| 2-stage Normal  | Normal(100,15)| [2, 1]                  | [1, 1]                 | [1,1]       | 15 | 166.2705   |
| 5-stage Normal  | Normal(32,5.657) | [3.5,2.5,1.5,1.0,0.5] | [1,1,0.5,0.5,0.5]      | [1,1,1,1,1] | 12 | 225.8672   |
| 5-stage Poisson | Poisson(32)   | [3.5,2.5,1.5,1.0,0.5]   | [1,1,0.5,0.5,0.5]      | [1,1,1,1,1] | 12 | 226.8458   |

Convention note: installation (local) holding is highest at the most-downstream stage (value added
downstream), so the env `holding_cost` (downstream->upstream) is decreasing; the echelon holding
costs the solver consumes are the installation-cost differences and are positive. The 2-stage matches
`stockpyl problem_6_1`; the 5-stage Normal/Poisson are `stockpyl problem_6_2a`/`problem_6_2b` scaled
x0.5 and time-rescaled to unit lead time (mean 64->32, penalty 24->12, holding x0.5). The exact
Poisson optimum is exposed to Python via `multi_echelon_serial_exact_poisson_solution` (alongside the
Normal `multi_echelon_serial_exact_normal_solution`), and the soft-tree rollout now accepts
`demand_kind="normal"|"poisson"` (`rollout.rs`, `bindings.rs`) so the Poisson instance trains under
the same protocol with discrete-count demand.

### Independent re-verification (2026-05-31)

The published numbers AND the `stockpyl.ssm_serial` reference values that `exact.rs` claims to
reproduce were re-confirmed independently of this repo, by calling `stockpyl 1.0.2`
`optimize_base_stock_levels` directly (with a small runtime numpy-2.x shim — see *Tooling note*):
Example 6.1 Normal → **C\* = 47.6654, S\* = {6.484, 12.028, 22.72}**; Poisson N=1/2/3 →
**4.220849 / 16.797779 / 72.043543** with **S\* = 8 / [7,13] / [9,15,26]** — i.e. exactly the
values asserted in `exact.rs`. The env transition/cost was also re-implemented from scratch in
Python (faithful port of `consume`/`replenish` + the echelon base-stock evaluator) and reproduces
the optima to ≤0.23% (Poisson) and exactly (Normal, continuous demand). The Rust verification tests
in `verification.rs` could not be executed here (no `cargo test` in this environment); the above is
the independent cross-check standing in for them.

### Caveat 1 — RESOLVED: Normal-demand evaluator now samples continuous demand

Previously `echelon_base_stock.rs::simulate` rounded Normal demand to an integer
(`normal.sample(...).round().max(0.0)`) while `exact.rs` optimizes against the *continuous* Normal;
that rounding biased the simulated Ex6.1 cost up to ≈48.44 (+1.62%) and the `verification.rs` Ex6.1
test passed only under a loose 2% tolerance. **Fixed (2026-06-04):** the Normal branch now samples
continuous demand (no `.round()`), so the env-sim reproduces the published optimum to **≈47.68
(+0.06%)** and the Ex6.1 env-sim assertion is tightened to 0.5% (`rel_err < 0.005`). The
env-dynamics — not just the exact solver — now reproduce the published 47.65. `consume`/`replenish`
never imposed rounding; this was purely an evaluator-sampling artifact.

### Caveat 2 — demand-facing lead time must be 1 (env under-counts when L₀ ≥ 2)

Carried verification instances all have most-downstream lead time = 1. The env was independently
confirmed to **under-count** cost when the demand-facing stage has lead time ≥ 2 (e.g. 2-stage,
downstream L=2: sim **≈20.1 vs exact ≈25.1, ≈20% under**), exactly as the `env.rs` docstring warns
(`env.rs:40–47`). Single-stage is correct at every lead time (L=1/2/3 all ≤0.18%). **Mechanism:**
the env charges installation holding on *physical on-hand only* and does **not** charge the
downstream echelon's in-transit pipeline; with L₀ ≥ 2 the optimal policy keeps inventory in transit
that the Clark-Scarf cost charges echelon holding on but the env does not, so the time-average
holding is undercharged. This is a cost-convention (in-transit accounting) issue, not a one-liner,
and is deferred with a proposal in `next_steps`.

## Benchmark (policies vs the exact optimum)

Instance set = the carried verification instances (all downstream L=1). Comparison of the optimal
echelon base-stock policy against two base-stock heuristics, evaluated on the env (Monte-Carlo,
400k periods, 5k warm-up, seeds {3,17,21}). Reproduce with
`scripts/multi_echelon_serial/benchmark_serial_clark_scarf.py`.

| instance       | exact C\* (published) | OPTIMAL (gap)        | newsvendor-per-echelon (gap) | lead-time-mean (gap) |
|----------------|-----------------------|----------------------|------------------------------|----------------------|
| Poisson N=1    | 4.2211 (4.220849)     | 4.2113 (−0.23%)      | 4.2113 (−0.23%)              | 8.7708 (+107.8%)     |
| Poisson N=2    | 16.7983 (16.797779)   | 16.7769 (−0.13%)     | 17.1887 (+2.32%)             | 22.7875 (+35.7%)     |
| Poisson N=3    | 72.0467 (72.043543)   | 72.0070 (−0.06%)     | 77.3686 (+7.39%)             | 123.94 (+72.0%)      |
| Normal Ex6.1   | 47.6654 (47.65)       | 48.4374 (+1.62%) †   | 49.8844 (+4.66%)             | 72.27 (+51.6%)       |

† Normal OPTIMAL gap is the demand-rounding bias (Caveat 1); with continuous demand it is +0.01%.

Reading: the optimal Clark-Scarf echelon base-stock policy reproduces the exact optimum to within
Monte-Carlo error; a per-echelon newsvendor that ignores the Clark-Scarf induced-penalty coupling
loses 2–7% on the multi-stage instances; the naive no-safety-stock lead-time-mean policy loses
36–120%. The exact optimum is the reference floor for any future learned policy.

**Learned soft-tree comparison — BLOCKED (no rebuild allowed).** The serial env is *not exposed to
Python*: there is no `serial_*` function in the installed `invman_rust`, `serial/bindings.rs` does
not exist, and `serial` is not registered in `multi_echelon/bindings.rs` (the `multi_echelon_*`
Python functions belong to `production_assembly_distribution_network`). A learned soft-tree rollout
on this env therefore cannot be run without adding a binding and rebuilding Rust. The benchmark
script is written so the trained policy drops straight in once the binding exists; the exact
blocker and the proposed binding are recorded in `next_steps`.

### Tooling note

`stockpyl 1.0.2` is incompatible with `numpy ≥ 2` in two spots (`helpers.py:348`
`np.array(..., copy=False)` and `ssm_serial.py:425` relying on numpy-1.x array-index squeeze). The
re-verification used a *runtime-only* shim (rewrites `copy=False`→`copy=None` and returns a scalar
index from `find_nearest`); no installed package files were modified.

## References

All five references below were independently re-verified against authoritative sources on
2026-05-31 (DOIs resolved at doi.org; publisher pages at pubsonline.informs.org / wiley.com;
stockpyl at readthedocs / PyPI / github.com/LarrySnyder).

- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475-490. DOI 10.1287/mnsc.6.4.475. (Verified at
  https://pubsonline.informs.org/doi/10.1287/mnsc.6.4.475 — venue/volume/issue/pages confirmed.)
- Federgruen, A., and P. Zipkin (1984). "Computational Issues in an Infinite-Horizon, Multiechelon
  Inventory Model." *Operations Research* 32(4):818-836. DOI 10.1287/opre.32.4.818. (Verified at
  https://pubsonline.informs.org/doi/10.1287/opre.32.4.818.)
- Chen, F., and Y.-S. Zheng (1994). "Lower Bounds for Multi-Echelon Stochastic Inventory Systems."
  *Management Science* 40(11):1426-1443. DOI 10.1287/mnsc.40.11.1426. (Verified at
  https://pubsonline.informs.org/doi/10.1287/mnsc.40.11.1426. Authors: Fangruo Chen, Yu-Sheng
  Zheng. stockpyl's exact serial recursion follows Chen & Zheng's reworking of Clark & Scarf.)
- Snyder, L. V., and Z.-J. M. Shen. *Fundamentals of Supply Chain Theory* (2nd ed., Wiley 2019),
  Example 6.1. ISBN 978-1-119-02484-2; book DOI 10.1002/9781119584445. (Verified at
  https://www.wiley.com/en-us/Fundamentals+of+Supply+Chain+Theory,+2nd+Edition-p-9781119024842.
  Example 6.1 published optimal cost ≈47.65 — corroborated by the author's `stockpyl` reference
  implementation, which loads it as `example_6_1` and reports `C* = 47.6687`; the textbook page
  itself is paywalled and was not read directly.)
- `stockpyl` (Snyder), `stockpyl.ssm_serial.optimize_base_stock_levels` — public reference
  implementation accompanying the textbook. https://stockpyl.readthedocs.io ;
  https://github.com/LarrySnyder/stockpyl ; PyPI `stockpyl`. Described in Snyder, L. V. (2023),
  "Stockpyl: A Python Package for Inventory Optimization and Simulation," *INFORMS Tutorials in
  Operations Research*, pp. 156-197, DOI 10.1287/educ.2023.0256. (Package + `ssm_serial` module +
  `example_6_1` instance verified present on 2026-05-31.)
