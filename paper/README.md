# Paper Workspace

This directory contains the new manuscript material for the problem extensions beyond the original
lost-sales paper.

Current scope:

- fixed-order-cost lost sales

Main files:

- [fixed_order_cost_lost_sales.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/fixed_order_cost_lost_sales.tex)
- [references.bib](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/references.bib)

The fixed-cost manuscript is kept self-contained. The benchmark results tables are now maintained
directly inside `fixed_order_cost_lost_sales.tex` instead of being generated through separate TeX
partials.

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
