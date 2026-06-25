# Paper Workspace

This directory contains the self-contained manuscript for the compact-CMA-ES inventory-control
benchmark study.

Current scope:

- vanilla / fixed-order-cost lost sales
- dual sourcing
- divergent multi-echelon (one-warehouse, K-retailer with special delivery; Van Roy 1997 /
  Gijsbrechts 2022)
- perishable inventory
- general-network backorder
- serial multi-echelon / Clark-Scarf
- one-warehouse multi-retailer
- ameliorating inventory
- production / assembly / distribution networks

Main files:

- [learning_inventory_control_policies_es.tex](learning_inventory_control_policies_es.tex)
- [references.bib](references.bib)

The manuscript is kept self-contained. The benchmark results tables are maintained directly inside
`learning_inventory_control_policies_es.tex` instead of being generated through separate TeX partials.

The paper uses comparator-specific verdicts rather than a single leaderboard: beat a heuristic only
under the same protocol, match proven optima, and report bound gaps or gate-matches explicitly. Recent
updates include the seed-robust OWMR gate beat on Kaynov instance 12, the mixed
production/assembly/distribution correction from a fragile flow-head audit to a residual
base-stock-backbone own-heuristic beat, and the
ImageNet-style baseline-ledger direction described in `../plan/README.md`.

The divergent multi-echelon section reports the learned soft-tree policy improving on the
in-environment constant base-stock by about 14.4% on both Gijs settings, with the published A3C
savings (8.95% and 12.09%) listed as cross-protocol context; env validation against the published
Van Roy constant base-stock costs is in the appendix. Reproduce the legacy multi-echelon results with:

```bash
python scripts/multi_echelon/train_multi_echelon_policy.py --reference gijsbrechts2022_setting1 --budget full
python scripts/multi_echelon/train_multi_echelon_policy.py --reference gijsbrechts2022_setting2 --budget full
```

To validate the literature anchor on the known-optimum instance:

```bash
python scripts/lost_sales_fixed_order_cost/validate_known_optimum.py
```

To rebuild the canonical benchmark suite first, if needed:

```bash
python scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py --reuse_existing --reuse_existing_summary
```

To rebuild the full-grid benchmark suite first, if needed:

```bash
python scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py --reuse_existing --reuse_existing_instance_summary
```

To push the benchmark manuscript to Overleaf:

```bash
python paper/push_to_overleaf.py --dry-run
python paper/push_to_overleaf.py
```

The push script uploads `paper/learning_inventory_control_policies_es.tex` to
`learning_inventory_control_policies_es.tex` (project root) in the Overleaf project
`invman_paper (revision)`. Authentication is handled by the local Overleaf session helper under
`/home/nima/code/tools/security/access/login_sessions/overleaf`.
