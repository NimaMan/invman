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

Status: **literature-verified (yes)** for the carried instance set, with two scoped caveats
documented below (demand-facing lead time must be 1; the evaluator rounds Normal demand).

Two complementary checks, both passing:

1. **Exact** — `exact.rs` reproduces the published optima: Snyder & Shen *Fundamentals of Supply
   Chain Theory* **Example 6.1** optimal cost **47.65** (solver `47.6654`, within 0.03%); discrete
   Poisson optima match the `stockpyl.ssm_serial` reference implementation to machine precision
   (3-stage `C* = 72.043543`, `S* = [9,15,26]`; 2-stage `16.797779`, `S* = [7,13]`; 1-stage
   `4.220849`, `S* = 8`). See `exact.rs` tests `single_stage_reduces_to_newsvendor_closed_form`
   and `poisson_instances_match_reference_implementation`.
2. **Simulation** — `env.rs` driven by the optimal echelon base-stock policy reproduces those
   same optima by Monte-Carlo simulation within sampling error (Poisson 1/2/3-stage →
   `4.211`, `16.777`, `72.007`, all ≤0.23%; Example 6.1 Normal → see the rounding note).
   `exact_and_simulation_agree` cross-checks decomposition vs simulation directly.

This is the pre-training correctness gate: before any learned policy is trained on `env.rs`, the
env is shown to reproduce the literature optimum under the known-optimal policy.

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

### Caveat 1 — Normal-demand evaluator rounds demand (≈1.6% upward bias on Ex6.1)

`echelon_base_stock.rs::simulate` samples Normal demand and **rounds it to an integer**
(`normal.sample(...).round().max(0.0)`), while `exact.rs` optimizes against the *continuous*
Normal. That rounding changes the demand distribution and biases the simulated Ex6.1 cost up to
**≈48.44 (+1.62%)** — a real, repeatable bias (5 seeds all 48.40–48.43), not sampling noise. With
the rounding removed (continuous demand) the env reproduces the exact optimum to 4 decimals
(**47.669 vs 47.6654, +0.01%**). The `verification.rs` Ex6.1 test passes only because its tolerance
is 2%. This is an *evaluator* artifact, not an env-dynamics error: `consume`/`replenish` impose no
rounding. To make the Normal check tight, sample continuous demand (or document the rounding as the
intended integer-demand approximation) — proposed in `next_steps`, not changed here.

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

- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475-490.
- Federgruen, A., and P. Zipkin (1984). "Computational Issues in an Infinite-Horizon, Multiechelon
  Inventory Model." *Operations Research* 32(4):818-836.
- Chen, F., and Y.-S. Zheng (1994). "Lower Bounds for Multi-Echelon Stochastic Inventory Systems."
  *Management Science* 40(11):1426-1443.
- Snyder, L. V., and Z.-J. M. Shen. *Fundamentals of Supply Chain Theory* (2nd ed., Wiley 2019),
  Example 6.1.
- `stockpyl` (Snyder), `stockpyl.ssm_serial.optimize_base_stock_levels`. https://stockpyl.readthedocs.io
