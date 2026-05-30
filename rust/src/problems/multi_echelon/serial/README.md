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

## Verification (env reproduces the literature)

Two complementary checks, both passing:

1. **Exact** — `exact.rs` reproduces the published optima: Snyder & Shen *Fundamentals of Supply
   Chain Theory* **Example 6.1** optimal cost **47.65** (within 0.05%); discrete Poisson optima
   match the `stockpyl.ssm_serial` reference implementation to machine precision (e.g. 3-stage
   `C* = 72.0435`, `S* = [9,15,26]`; 2-stage `16.7978`; 1-stage `4.2208`).
2. **Simulation** — `env.rs` driven by the optimal echelon base-stock policy reproduces those
   same optima by Monte-Carlo simulation within sampling error (Example 6.1 → ≈47.6; Poisson
   3-stage → ≈72.1). `exact_and_simulation_agree` cross-checks decomposition vs simulation
   directly.

This is the pre-training correctness gate: before any learned policy is trained on `env.rs`, the
env is shown to reproduce the literature optimum under the known-optimal policy.

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
