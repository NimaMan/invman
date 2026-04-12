# Multi-Echelon

This folder contains multiple multi-echelon problem formulations that should stay separate because
they do not share the same dynamics or benchmark contract.

## Subproblems

- `divergent_special_delivery/`
  - Van Roy / Gijs one-warehouse-multi-retailer family with same-day special delivery
- `general_backorder_fixed_cost/`
  - Geevers/CardBoard Company general-network family with backorders and unit lead times

## Verification Status

At the moment, none of the multi-echelon subproblems in this folder are literature-verified.

- `divergent_special_delivery/`
  - literature benchmark rows are carried from Van Roy and Gijs
  - the current repo implementation does not reproduce those rows tightly enough to claim
    literature verification
- `general_backorder_fixed_cost/`
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
