# Agent Guide

This repository is meant to be runnable by an agent from the repo root:

- repo root: `/path/to/invman`
- all commands below assume `cwd = repo root`

Use the curated experiment layer first. The main entrypoint is:

- `python numerical_experiments/run.py --list`

Do not guess script names or policy ids when a suite already exists in the catalog.

## Environment Setup

Create and activate a virtualenv in the repo root or in the parent directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install maturin
python scripts/build_rust_extension.py
```

The Rust build helper is portable across these layouts:

- `.venv/` inside the repo
- `.venv/` in the parent directory
- the currently active interpreter via `sys.executable`

If Rust source changes, rebuild the extension:

```bash
python scripts/build_rust_extension.py
```

## Sanity Checks

Run these before long experiments on a fresh machine:

```bash
python -m pytest tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py tests/test_numerical_experiments_catalog.py -q
python numerical_experiments/run.py --list
```

For a broader health check:

```bash
python -m pytest tests -q
```

## Ready Experiment Suites

List stable suites:

```bash
python numerical_experiments/run.py --list --status ready
```

Run the canonical fixed-cost benchmark:

```bash
python numerical_experiments/run.py --suite fixed_cost_single_instance_check
```

Run the canonical vanilla lost-sales benchmark:

```bash
python numerical_experiments/run.py --suite lost_sales_single_instance_check
```

Run the full vanilla lost-sales paper-style grid:

```bash
python numerical_experiments/run.py --suite lost_sales_full_policy_grid
```

Run the full fixed-cost paper-style grid:

```bash
python numerical_experiments/run.py --suite fixed_cost_full_policy_grid
```

Run a single suite without executing it:

```bash
python numerical_experiments/run.py --suite fixed_cost_full_policy_grid --dry-run
```

Run all stable suites:

```bash
python numerical_experiments/run.py --all-ready
```

## Fixed-Cost Paper Workflow

The fixed-order-cost lost-sales section is the most mature new problem family.

Main scripts:

- single-instance preflight: `scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py`
- full 16-instance literature-aligned grid: `scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`
- paper table export: `scripts/lost_sales_fixed_order_cost/export_paper_table.py`

Canonical benchmark outputs:

- `outputs/benchmarks/fixed_cost_l4_canonical_suite_5k_paperlike/`

Full grid outputs:

- `outputs/benchmarks/fixed_cost_full_grid_suite_5k_paperlike/`

Use the single-instance preflight first to confirm that the full experiment path is behaving as expected.

Refresh the canonical paper table after the benchmark exists:

```bash
python scripts/lost_sales_fixed_order_cost/export_paper_table.py
```

Current fixed-cost policy set:

- `linear_categorical_quantity`
- `linear_gated_ordinal_quantity`
- `nn_categorical_quantity`
- `nn_gated_ordinal_quantity`
- `soft_tree_depth2_linear_leaf`
- `soft_tree_depth1_linear_leaf`

Current heuristic set:

- `s_s`
- `s_nq`
- `modified_s_s_q`

Important note:

- `nn_categorical_quantity` is still marked provisional on the canonical fixed-cost benchmark because it matched the linear categorical baseline exactly and should be re-verified before publication claims rely on it.

## Outputs

Ad hoc experiment runs write to:

- `outputs/results/`
- `outputs/logs/`
- `outputs/models/`

Benchmark suites write to:

- `outputs/benchmarks/<run_tag>/`

For the fixed-cost full-grid suite, each instance also gets:

- `outputs/benchmarks/<run_tag>/instances/<reference>.json`

These per-instance JSON files contain:

- reference parameters
- literature metadata
- heuristic search parameters and evaluations
- learned-policy result paths
- comparative summaries vs the best heuristic

## Paper Files

The new manuscript workspace is:

- `paper/fixed_order_cost_lost_sales.tex`
- `paper/generated/fixed_cost_canonical_table.tex`
- `paper/references.bib`

LaTeX is optional for experiments. A TeX toolchain is not required to run benchmarks.

## Linux Notes

On a separate Linux machine:

- install Rust toolchain and a C toolchain before building the extension
- keep the commands above unchanged after activating the repo virtualenv
- prefer `--reuse_existing` or `--reuse_existing_instance_summary` on reruns of expensive benchmark scripts
- increase `--mp_num_processors` if the machine has more cores

## Source of Truth

When in doubt, use these files:

- experiment catalog: `numerical_experiments/catalog.py`
- lost-sales experiment definitions: `invman/problems/lost_sales/experiment_spec.py`
- lost-sales instance registry: `invman/problems/lost_sales/reference_instances.py`
- fixed-cost experiment definitions: `invman/problems/lost_sales_fixed_order_cost/experiment_spec.py`
- fixed-cost instance registry: `invman/problems/lost_sales_fixed_order_cost/reference_instances.py`
- core runner: `invman/experiment_runner.py`
