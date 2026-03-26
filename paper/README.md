# Paper Workspace

This directory contains the new manuscript material for the problem extensions beyond the original
lost-sales paper.

Current scope:

- fixed-order-cost lost sales

Main files:

- [fixed_order_cost_lost_sales.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/fixed_order_cost_lost_sales.tex)
- [references.bib](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/references.bib)
- [generated/fixed_cost_canonical_table.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/generated/fixed_cost_canonical_table.tex)

To refresh the canonical fixed-cost results table from the benchmark JSON:

```bash
python scripts/export_fixed_cost_paper_table.py
```

To rebuild the canonical benchmark suite first, if needed:

```bash
python scripts/benchmark_fixed_cost_canonical_suite.py --reuse_existing --reuse_existing_summary
```
