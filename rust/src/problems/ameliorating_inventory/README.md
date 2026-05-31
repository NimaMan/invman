# ameliorating_inventory

Rust-first problem home for `ameliorating_inventory`.

Source paper: Pahr and Grunow (2025), *The Value of Blending — Managing Ameliorating Inventory
Using Deep Reinforcement Learning*, Production and Operations Management
(DOI `10.1177/10591478251387795`; companion code
`github.com/amelioratinginventory/ameliorating_inventory`).

## Problem in the source paper

- age-structured inventory of products that *ameliorate* (gain value) while stored
  (whiskey, port wine, cheese)
- multiple age-differentiated products, each with a target age
- decisions: purchasing volume, production volume per product, and issuance per age
- products may be **blended** across ages, giving operational flexibility
- stochastic purchase prices, stochastic sales prices (copula-correlated with demand),
  stochastic age-dependent decay, evaporation, and a processing-capacity limit
- objective: maximize long-run **average profit**; performance reported as the gap to a
  perfect-information LP upper bound

## Current Rust interpretation (reduced)

- discrete, finite-horizon, **discounted-cost** reduction
- single decision: purchase quantity into the youngest age class
- amelioration: surviving units age up one class per period (oldest class absorbing)
- exact average-age blending issuance: a product ships only if the issued blend's mean age is
  at least the product target age; young and old stock may be combined to reach an older target
- fixed product prices, fixed purchase cost, fixed age-retention, fixed decay-salvage values
- repo-native exact finite-horizon DP verifier on a small instance

## Current status: NOT literature-verified (repo-exact verified only)

The env is internally self-consistent (the worked transition reproduces exactly via the installed
binding; the exact DP dominates both carried heuristics in `verification/tests.rs`), but it is a
reduced approximation of the paper, not a faithful port. No published number anchors any
executable assertion.

### Root cause of the gap (structural, not a bug)

See `literature/README.md` for the term-by-term table. The decisive gaps: discounted-cost vs.
average-profit objective; purchase-only action vs. the paper's purchase + production + issuance
action; fixed prices/decay vs. stochastic purchase/sales prices and stochastic beta decay; no
processing-capacity constraint; 5 ages / 2 products vs. the companion default 10 ages / 3 products
(or 25 ages for the port-wine case study). These cannot be closed by localized edits to the
present env; they require a new env.

## Benchmark situation

- Feasible now (without rebuilding Rust): heuristics-vs-heuristics and learned-soft-tree-vs-heuristics
  on the repo-native instances, via the installed bindings
  (`ameliorating_inventory_simulate_policy`, `..._policy_rollout_from_paths`,
  `..._soft_tree_rollout*`). These produce internally meaningful numbers but are **not**
  literature-comparable, because the instances and objective are repo-native.
- A sanity benchmark on `PRIMARY_REFERENCE_INSTANCE` (5 ages, 2 products, Poisson [10,6], 40
  periods, 2000 reps, γ=0.99) gives discounted profit ≈ 10,444 (newsvendor_purchase, target 24)
  and ≈ 10,468 (two_dimensional_order_up_to, targets 24/8, cutoff 1). These are illustrative only.
- Blocked: the exact-optimal DP (`finite_horizon_dp::solve_optimal_policy`) is `#[cfg(test)]`
  and has **no Python binding**, so optimal-vs-heuristic-vs-learned benchmarking from Python is not
  possible without a new binding (see the package functionality notes / next steps).
- Not literature-comparable at all until the executable formulation gap is closed.

## Package layout

- literature references and recorded anchors: `literature/references.rs`, `literature/README.md`
- verification code: `verification/tests.rs`
- exact reduced solver (test-only, no binding): `finite_horizon_dp.rs`
- env transition + cost + blending issuance: `env.rs`, `issuance.rs`
- demand models: `demand.rs`
- heuristics: `heuristics/`
- learned-policy rollout and bindings: `rollout.rs`, `bindings.rs`
- practical notes: `practical/`
- experiment notes: `experiments/`
