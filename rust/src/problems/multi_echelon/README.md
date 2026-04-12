# Multi-Echelon

This folder contains multiple multi-echelon problem formulations that should stay separate because
they do not share the same dynamics or benchmark contract.

## Subproblems

- `divergent_special_delivery/`
  - Van Roy / Gijs one-warehouse-multi-retailer family with same-day special delivery
- `general_backorder_fixed_cost/`
  - Geevers/CardBoard Company general-network family with backorders and unit lead times

## Structure Rule

The root `multi_echelon/` folder should only contain:

- formulation subfolders
- the root `mod.rs`
- the root `bindings.rs`
- this overview README

Any formulation-specific literature, verification, reports, or experiments belong inside the relevant
subproblem folder, not at the root.
