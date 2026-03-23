# Lost-Sales Autoresearch

This directory adapts the `karpathy/autoresearch` idea to the inventory-management repo.

Primary references:

- repository: https://github.com/karpathy/autoresearch
- agent program: https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md

The first target is the trusted vanilla lost-sales benchmark:

- reference instance: `vanilla_l4_p4_poisson5`
- learned metric: mean long-run cost after warm-up
- baseline heuristics: Myopic-1, Myopic-2, SVBS

Unlike Karpathy's single-GPU training setup, the initial inventory-management loop uses a fixed
simulation budget instead of a fixed wall-clock budget. This keeps policy quality comparisons fair
across policy classes while the Rust rollout path is still being expanded.

The idea we keep:

- one trusted benchmark
- one narrow editable surface
- one fixed experiment budget
- automatic result logging
- keep/discard decisions against a running baseline

The main adaptation:

- Karpathy fixes wall-clock time because all experiments share one training script on one GPU.
- Here we fix the rollout/training budget because policy families currently have different runtime
  backends (`python` vs `rust`) and we care first about policy quality on the benchmark.

Budgets are a default protocol, not a hard restriction. In practice we use:

- screening runs to reject weak ideas quickly
- promoted full-budget runs for promising architectures

Key files:

- `program_lost_sales.md`: agent instructions for the autonomous loop
- `../scripts/autoresearch_lost_sales.py`: fixed-budget experiment runner and logger
- `../scripts/autoresearch_tree_structures.py`: focused tree-structure comparison runner
- `../scripts/build_rust_extension.py`: helper to rebuild the Rust extension in the shared virtualenv

The second target is the canonical fixed-order-cost benchmark:

- reference instance: `lit_pois_mu5_l4_p4_k5`
- learned metric: mean long-run cost after warm-up
- baseline heuristics: `s,S`, `s,nQ`, modified `s,S,q`

Fixed-cost files:

- `program_fixed_order_cost.md`: agent instructions for the fixed-cost loop
- `../scripts/autoresearch_fixed_order_cost.py`: fixed-cost keep/discard runner and ledger
- `../scripts/autoresearch_fixed_order_tree_structures.py`: focused tree-structure comparison runner
- `../scripts/evaluate_saved_policy.py`: long-horizon reevaluation helper for promoted candidates

## Current fixed-cost result

Best fixed-cost autoresearch result so far:

- screening winner: oblique depth-1 linear-leaf tree, temperature `0.25`, sigma `5.0`
- full-budget `50k` evaluation: `8.77528`
- long-horizon `1M` reevaluation: `8.76576`

This improved on:

- the earlier transferred depth-2 fixed-cost tree: `8.81009`
- the best heuristic baseline on the canonical instance: `9.16537`

## Current result

Best tree architecture found so far on the trusted vanilla benchmark:

- `soft_tree_oblique_tree_linear_leaf_quantity_pipeline`
- depth `2`
- mean cost: `4.753725`

This came from two stages:

- tree split-structure screening showed `oblique` was better than `axis_aligned`
- full-budget leaf comparison showed `linear` leaves were much better than `constant` leaves

The next targets are the two additional Gijsbrechts (2022) problem classes:

- dual sourcing: `program_dual_sourcing.md`, `../scripts/autoresearch_dual_sourcing.py`
- multi echelon: `program_multi_echelon.md`, `../scripts/autoresearch_multi_echelon.py`

Current smoke results:

- dual sourcing primary instance (`lr=4`, `ce=110`): learned tree `249.84`, best heuristic `220.73`
- multi-echelon setting 2: learned tree `3776.45`, best constant base-stock benchmark `3776.45`

## Dual-Sourcing status

The dual-sourcing smoke result was not the final word. A full-budget rerun on the same primary instance
now gives:

- learned oblique depth-2 linear-leaf tree: `233.08375`
- best heuristic baseline, capped dual-index: `221.61`

That is a real improvement over the smoke run, but still leaves the learned tree about `5.2%` behind the
best heuristic.

So the current dual-sourcing conclusion is:

- the dual-sourcing environment, heuristic search, DP benchmark, and Rust rollout path are working;
- additional CMA-ES budget helps;
- but the next high-value change is a better policy search space, not just more training.

The most motivated next autoresearch target is a state-dependent target-position policy for dual sourcing,
because the benchmark heuristics act on expedited and regular inventory positions rather than directly on
raw `(q_regular, q_expedited)` orders.
