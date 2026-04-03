# Numerical Experiments

This folder is the experiment catalog and launch layer for the project.

Its purpose is to keep one explicit place that answers:

- which problem sets are part of the paper-style numerical experiments
- which heuristics must be included for each problem
- which learned policy families must be included for each problem
- which suites are already stable enough to run on a separate Linux machine
- which suites are still exploratory and meant for policy-design work

## Structure

- [catalog.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/numerical_experiments/catalog.py):
  curated suite registry
- [run.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/numerical_experiments/run.py):
  list and run suites from the catalog

The runner does not re-implement the experiment logic. It delegates to the benchmark and
autoresearch scripts already in [scripts](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/scripts).
That keeps the catalog clean while avoiding duplicate experiment code.

## Problem Matrix

### Vanilla lost sales

Heuristics:

- `myopic1`
- `myopic2`
- `svbs`
- literature `capped_base_stock`
- literature `optimal` when reported

Policy families:

- base:
  - `linear_categorical_quantity_q8`
  - `linear_categorical_quantity_q20`
  - `nn_categorical_quantity_q8`
  - `nn_categorical_quantity_q20`
- improved:
  - `soft_tree_depth2_linear_leaf_q8`

Current interpretation:

- the stable setup is intentionally split into only two modes:
  - one single-instance preflight run
  - one full literature-aligned grid run
- the single-instance run is only for checking that the full experiment path behaves as intended
- the full policy grid suite is the data-generation path for paper tables

Status:

- single-instance preflight suite is ready
- full policy grid suite is ready

### Fixed-order-cost lost sales

Heuristics:

- `s_s`
- `s_nq`
- `modified_s_s_q`

Policy families:

- base:
  - `linear_categorical_quantity`
  - `nn_categorical_quantity`
- improved:
  - `linear_soft_gated_ordinal_quantity`
  - `nn_soft_gated_ordinal_quantity`
  - `soft_tree_depth2_linear_leaf`
  - `soft_tree_depth1_linear_leaf`

Current interpretation:

- this problem already shows that action-space design matters strongly
- the stable setup is intentionally split into only two modes:
  - one single-instance preflight run
  - one full literature-aligned grid run
- the single-instance run is only for checking that the full experiment path behaves as intended
- the full policy grid suite is the data-generation path for paper tables

Status:

- single-instance preflight suite is ready
- full policy grid suite is ready

### Dual sourcing

Heuristics:

- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`
- `optimal_dp` on the small reference settings

Policy families:

- base:
  - `linear_bounded_quantity_identity`
  - `nn_bounded_quantity_identity`
  - `soft_tree_identity`
- improved / structured:
  - `linear_base_surge_targets`
  - `nn_base_surge_targets`
  - `soft_tree_base_surge_targets`

Current interpretation:

- this problem still requires policy-design work
- structured action spaces appear necessary

Status:

- exploratory

### Multi-echelon

Heuristics / references:

- `constant_base_stock`

Policy families:

- current:
  - `soft_tree_constant_leaf`
  - `soft_tree_linear_leaf`

Current interpretation:

- the benchmark layer exists
- the final learned policy family is not yet frozen

Status:

- exploratory

## Linux Usage

Build the environment and native extension on the Linux machine first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m pip install maturin
python scripts/build_rust_extension.py
```

List the suites:

```bash
python numerical_experiments/run.py --list
```

Dry-run all stable suites:

```bash
python numerical_experiments/run.py --all-ready --dry-run
```

Run one stable suite:

```bash
python numerical_experiments/run.py --suite lost_sales_single_instance_check
```

Run the full vanilla lost-sales grid suite:

```bash
python numerical_experiments/run.py --suite lost_sales_full_policy_grid
```

Run the full fixed-cost grid suite:

```bash
python numerical_experiments/run.py --suite fixed_cost_full_policy_grid
```

Run all stable suites:

```bash
python numerical_experiments/run.py --all-ready
```

Run exploratory suites explicitly:

```bash
python numerical_experiments/run.py --suite dual_sourcing_backbone_screen --suite multi_echelon_autoresearch
```

## Design Principle

The catalog is organized by problem type, not by code module.

For each problem, we want:

- a fixed benchmark problem set
- a fixed heuristic set
- a fixed learned-policy family set
- one preflight suite for a single instance
- one stable suite that can generate the paper-style tables over the full problem set

Exploratory suites are allowed, but they should be clearly labeled as exploratory until the policy
family for that problem is frozen.
