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

## Current Defaults

The live benchmark defaults have moved since the original paperlike `5k` runs.
When in doubt, trust the current experiment spec files over older run tags:

- vanilla lost sales:
  - [experiment_spec.py](/home/nima/code/ml/invman/invman/problems/lost_sales/experiment_spec.py)
- fixed-cost lost sales:
  - [experiment_spec.py](/home/nima/code/ml/invman/invman/problems/lost_sales_fixed_order_cost/experiment_spec.py)

Current defaults:

- `training_episodes = 2000`
- `es_population = 64`
- `training_horizon = 2000`
- `eval_horizon = 1e6`
- `eval_seeds = 10`

The active paper-style lost-sales grids currently exclude `L=2`; the reported lead-time axis is
`L in {4, 6, 8, 10}`.

## Canonical Single Instances

If you need one representative instance before launching a larger sweep, use:

- vanilla lost sales:
  - `vanilla_l4_p4_poisson5`
- fixed-cost lost sales:
  - `lit_pois_mu5_l4_p4_k5`

These are the default single-instance references used throughout the current architecture work.

## Fixed-Cost Paper Workflow

The fixed-order-cost lost-sales section is still the most mature extension problem family, but the
active protocol is now the shorter `2k / pop64 / h2000` setup, not the older `5k paperlike`
workflow.

Main scripts:

- known-optimum heuristic validation: `scripts/lost_sales_fixed_order_cost/validate_known_optimum.py`
- single-instance preflight: `scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py`
- full 16-instance literature-aligned grid: `scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`

Use the single-instance preflight first to confirm that the full experiment path is behaving as expected.

Current fixed-cost policy set:

- `linear_categorical_quantity`
- `linear_soft_gated_ordinal_quantity`
- `nn_categorical_quantity`
- `nn_soft_gated_ordinal_quantity`
- `soft_tree_depth2_linear_leaf`
- `soft_tree_depth1_linear_leaf`

Current heuristic set:

- `s_s`
- `s_nq`
- `modified_s_s_q`

Important note:

- `nn_categorical_quantity` is still marked provisional on the canonical fixed-cost benchmark because it matched the linear categorical baseline exactly and should be re-verified before publication claims rely on it.

## Current Architecture Work

The active architecture note is:

- [README.md](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/README.md)

That note is the current source of truth for:

- direct-head geometry experiments
- vanilla vs fixed-cost comparisons
- Poisson / Geometric / correlated-demand single-instance sweeps
- the published fixed-cost validation anchor from Bijvank, Bhulai, and Huh (2015)

The current linear direct/tree family under study is:

- `linear_categorical_quantity`
- `linear_sigmoid_direct_quantity`
- `linear_direct_quantity`
- `linear_gated_sigmoid_direct_quantity`
- `linear_soft_gated_direct_quantity`
- `linear_hard_gated_direct_quantity`
- `linear_soft_gated_ordinal_quantity`
- `soft_tree_depth1_linear_leaf`
- `soft_tree_depth2_linear_leaf`

The raw unbounded direct head is intentionally not part of the active experiment surface anymore.

## Demand Families

Lost-sales demand families currently implemented are:

- `Poisson`
- `Geometric`
- `MarkovModulatedPoisson2`

The current correlated-demand sweeps use two mean-preserving `MarkovModulatedPoisson2` settings:

- positive correlation:
  - `lambda_low = 3`
  - `lambda_high = 7`
  - `p00 = p11 = 0.9`
- negative correlation:
  - `lambda_low = 3`
  - `lambda_high = 7`
  - `p00 = p11 = 0.1`

The Rust implementation is:

- [mod.rs](/home/nima/code/ml/invman/rust/src/problems/lost_sales/demand/mod.rs)

## Outputs

Ad hoc experiment runs write to:

- `outputs/results/`
- `outputs/logs/`
- `outputs/models/`

Benchmark suites write to:

- `outputs/benchmarks/<run_tag>/`

Runs launched through `run_experiment(...)` also write a status sidecar:

- `outputs/benchmarks/<run_tag>/results/status_<experiment_name>.json`

Use these files to determine whether a run:

- completed normally
- is still training
- is evaluating
- was interrupted
- failed with an exception

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

- `paper/learning_inventory_control_policies_es.tex`
- `paper/references.bib`

The TeX file is currently a working note aligned with the architecture study, not just the old
fixed-cost-only paper snapshot. If you update the manuscript, also check:

- [README.md](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/README.md)

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
