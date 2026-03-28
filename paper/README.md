# Paper Workspace

This directory contains the new manuscript material for the problem extensions beyond the original
lost-sales paper.

Current scope:

- fixed-order-cost lost sales

Main files:

- [fixed_order_cost_lost_sales.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/fixed_order_cost_lost_sales.tex)
- [references.bib](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/references.bib)
- [generated/fixed_cost_canonical_table.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/generated/fixed_cost_canonical_table.tex)
- [generated/fixed_cost_full_grid_table.tex](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/paper/generated/fixed_cost_full_grid_table.tex)

To refresh the canonical fixed-cost results table from the benchmark JSON:

```bash
python scripts/lost_sales_fixed_order_cost/export_paper_table.py
```

To refresh the full-grid fixed-cost results table from the benchmark JSON:

```bash
python scripts/lost_sales_fixed_order_cost/export_full_grid_paper_table.py
```

To rebuild the canonical benchmark suite first, if needed:

```bash
python scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py --reuse_existing --reuse_existing_summary
```

To rebuild the full-grid benchmark suite first, if needed:

```bash
python scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py --reuse_existing --reuse_existing_instance_summary
```

To run the full-grid benchmark and refresh the paper table in one step:

```bash
python scripts/lost_sales_fixed_order_cost/run_full_grid_paper_job.py --mp_num_processors 16 --instance_jobs 4
```
