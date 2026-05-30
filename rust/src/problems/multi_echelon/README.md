# Multi-Echelon

This folder contains multiple multi-echelon problem formulations that should stay separate because
they do not share the same dynamics or benchmark contract.

## Subproblems

Each subproblem is a distinct *version* of the multi-echelon problem (different topology and/or
contract). More versions can be added here as siblings.

- `serial/`
  - the textbook serial system (Clark & Scarf 1960): N stages in series, echelon base-stock optimal
- `general_network/`
  - the Pirhooshyaran & Snyder (2021) general acyclic supply network: raw + finished inventory,
    `single`/`assembly`/`distribution` nodes, pairwise order-up-to decisions (the most general
    topology here; was previously the top-level `network_inventory` family)
- `divergent_special_delivery/`
  - Van Roy / Gijs one-warehouse-multi-retailer family with same-day special delivery
- `general_backorder_fixed_cost/`
  - Geevers/CardBoard Company general-network family with backorders and unit lead times

## Verification Status

- `serial/` — **literature-verified**. The `env.rs` simulation under the optimal echelon
  base-stock policy reproduces the published optima (Snyder & Shen Example 6.1 cost 47.65; discrete
  Poisson 3-stage 72.04, 2-stage 16.80, 1-stage 4.22) within Monte-Carlo error, and the `exact`
  solver reproduces them analytically (within 0.05%, cross-checked against `stockpyl.ssm_serial`).
- `general_network/` — not literature-verified
  - implements the richer Pirhooshyaran model (per-node production step + pipeline holding), which
    does not reduce to the textbook serial/assembly optima; the paper's general-network simulation
    protocol could not be recovered from public sources. Single-node newsvendor rows are reproduced
    analytically. See its README and `serial_echelon_simulation.rs` for the structural gap.
- `divergent_special_delivery/` — not literature-verified
  - literature benchmark rows are carried from Van Roy and Gijs
  - the current repo implementation does not reproduce those rows tightly enough to claim
    literature verification
- `general_backorder_fixed_cost/` — not literature-verified
  - literature benchmark rows are carried from Geevers
  - set 1 is close, but the current repo implementation does not reproduce sets 2 and 3, so this
    formulation is also not literature-verified

## Structure Rule

The root `multi_echelon/` folder should only contain:

- formulation subfolders
- the root `mod.rs`
- the root `bindings.rs`
- this overview README

Any formulation-specific literature, verification, reports, or experiments belong inside the relevant
subproblem folder, not at the root.
