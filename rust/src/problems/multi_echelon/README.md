# Multi-Echelon

This folder is now an umbrella for multiple multi-echelon formulations.

These formulations are related, but they are not the same executable problem. They differ in network
structure, event timing, unmet-demand handling, cost terms, and benchmark heuristics, so they should
not share one `env.rs` and one `rollout.rs`.

## Subfamilies

- `divergent_special_delivery/`
  - the current Van Roy / Gijs family
  - one warehouse, multiple identical retailers
  - unmet store demand can trigger same-day special deliveries with probability `P_w`
  - current implementation lives here
- `general_backorder_fixed_cost/`
  - planned home for the Geevers-style family
  - general backorder network with fixed ordering costs
  - currently only a placeholder

## Current Status

- the active Rust and Python bindings exposed from `problems::multi_echelon` currently point to
  `divergent_special_delivery`
- `divergent_special_delivery` is not literature-verified yet
- `general_backorder_fixed_cost` has not been implemented yet

## Rule

When a multi-echelon formulation has different state semantics or cost structure, it gets its own
subfolder under `multi_echelon/` rather than being folded into a single shared runtime path.
