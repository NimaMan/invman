# Lost-Sales Autoresearch

This directory adapts the `karpathy/autoresearch` idea to the inventory-management repo.

> **Designing a policy for a new problem?** Read [`POLICY_DESIGN_GUIDELINES.md`](POLICY_DESIGN_GUIDELINES.md)
> first — the reusable recipe (anchor the env to published costs → treat the action
> parameterization as part of the policy → encode in the best heuristic's coordinate system →
> warm-start CMA-ES → autoresearch loop → checklist for adding a new problem to the paper).

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
- `../scripts/lost_sales/autoresearch_lost_sales.py`: fixed-budget experiment runner and logger
- `../scripts/autoresearch_tree_structures.py`: focused tree-structure comparison runner
- `../scripts/build_rust_extension.py`: helper to rebuild the Rust extension in the active virtualenv

The second target is the canonical fixed-order-cost benchmark:

- reference instance: `lit_pois_mu5_l4_p4_k5`
- learned metric: mean long-run cost after warm-up
- baseline heuristics: `s,S`, `s,nQ`, modified `s,S,q`

Fixed-cost files:

- `program_fixed_order_cost.md`: agent instructions for the fixed-cost loop
- `../scripts/autoresearch_fixed_order_cost.py`: fixed-cost keep/discard runner and ledger
- `../scripts/autoresearch_fixed_order_tree_structures.py`: focused tree-structure comparison runner
- `../scripts/evaluate_saved_policy.py`: long-horizon reevaluation helper for promoted candidates
- `fixed_cost_ordinal_stability/README.md`: focused note on when the ordinal fixed-cost head works,
  when it fails, and how the stored Rust/Python baselines differ

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

- dual sourcing: `program_dual_sourcing.md`, `dual_sourcing_policy_search/README.md`, `../scripts/dual_sourcing/autoresearch_dual_sourcing.py`
- multi echelon: `program_multi_echelon.md`, `../scripts/multi_echelon/autoresearch_multi_echelon.py`
- general-backorder-fixed-cost (Geevers set-1 general network): `program_general_backorder_fixed_cost.md`, `../scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py`. Learned depth-2 constant-leaf `node_base_stock_targets`, warm-started at the published levels `[82,100,64,83,35x5]`, beats the constant node-base-stock benchmark (repo `~10,355`, published `10,467`) by `-22.4%` (held-out `8,034.8 +/- 17.6`, 2,000 CRN seeds) and lands `679` below the published PPO best `8,714`; robust across two CMA seeds (`8,034.8` and `7,590.7`).

Current smoke results:

- dual sourcing primary instance (`lr=4`, `ce=110`): learned tree `249.84`, best heuristic `220.73`
- multi-echelon (superseded the old `3776.45` smoke): with the `direct_level` action design the
  learned soft tree beats the in-env best constant base-stock by ~14.4% on **both** faithful Gijs
  settings (setting 1: `779.8` vs `911.4`, > published A3C `8.95%`; setting 2: `973.6` vs `1137.8`,
  > published A3C `12.09%`). The old `3776.45` was an artifact of the Gijs reduced `{50..100}`
  action grid starving the warehouse. See `program_multi_echelon.md` for the search direction.

## Dual-Sourcing status

The dual-sourcing search has now moved beyond the old single-instance smoke framing.

The current question is:

- across the six Gijs Figure 9 rows, which policy-design factors matter most?

The dedicated workspace is now:

- `dual_sourcing_policy_search/README.md`

The current working hypothesis is:

- control geometry matters more than raw parameter count
- factorized dual-index controls are a better search space than raw direct-order outputs
- small discrete regular-order caps and tighter tree geometry matter most on the harder `l_r=3` and `l_r=4` rows
- the best current direction is row-dependent: axis-linear capped-delta variants for `l_r=2`, tighter axis-constant small-cap trees for `l_r in {3,4}`

## Losing-case autoresearch targets

After benchmarking learned soft-tree policies against the heuristics on the verified problems,
three problems where the learned policy does **not** yet beat the strongest heuristic each get a
dedicated single-policy autoresearch loop (program file + runner + TSV ledger), mirroring the
lost-sales / dual-sourcing / multi-echelon setup. Each runner trains one CLI-selected soft-tree on
a currently-losing instance and logs cost + gap vs the strongest heuristic on a held-out
common-random-number block; the keep/discard gate is the in-repo tuned heuristic.

- **one-warehouse multi-retailer** — `program_one_warehouse_multi_retailer.md`,
  `../scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py`. Losing the
  Kaynov (2024) instances to the tuned echelon base-stock + allocation by 0.43–1.69% (closest:
  partial-backorder `instance_11`, −0.43%). Levers: tree depth/temperature/split/leaf, action
  design (`symmetric_echelon_targets` vs `direct_orders`/`vector_quantity`), allocation policy,
  CMA-ES warm-start at the best base-stock.
- **joint replenishment** — `program_joint_replenishment.md`,
  `../scripts/joint_replenishment/autoresearch_joint_replenishment.py`. Losing MOQ on the 10
  high-cost van Vuchelen (2020) settings (worst: setting 4, −18.13%). Levers: tree structure, a
  base-stock-anchored action adapter, CMA-ES warm-start at MOQ, deeper budget on the high-cost
  losers.
- **vendor-managed inventory** — `program_vendor_managed_inventory.md`,
  `../scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py`. Losing/tying the
  tuned base-stock on ~4/5 reduced single-retailer instances (self-consistent env, no published
  anchor). Levers: tree structure, action design, CMA-ES warm-start at the base-stock control.
- **perishable inventory** — `program_perishable_inventory.md`,
  `../scripts/perishable_inventory/autoresearch_perishable_inventory.py`. On the two exact-verified
  De Moor (2022) / Farrington (2025) m=2/L=1 instances the warm-started depth-2 soft tree **beats**
  the in-repo base-stock gate by 1.16% (FIFO) / 0.82% (LIFO) at full budget, landing within noise of
  the analytic VI optimum. Lever: the softplus base-stock encoding of the scalar order head over the
  age-state + CMA-ES warm-start at the published best base-stock; validation-block model selection.
