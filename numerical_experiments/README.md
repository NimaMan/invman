# Numerical Experiments

This folder is the curated experiment-suite catalog and launch layer for the
project.

It intentionally stays separate from `scripts/`: scripts own the concrete
problem-specific training, validation, and reporting logic; this package owns the
cross-problem registry that says which scripts form a coherent numerical
experiment suite. Keeping that registry as a small importable package gives the
README, agent guide, and tests one stable command surface:
`python numerical_experiments/run.py ...`.

Its purpose is to keep one explicit place that answers:

- which problem sets are part of the paper-style numerical experiments
- which heuristics must be included for each problem
- which learned policy families must be included for each problem
- which suites are already stable enough to run on a separate Linux machine
- which suites are still exploratory and meant for policy-design work

## Structure

- `catalog.py`: curated suite registry.
- `run.py`: list and run suites from the catalog.
- `__init__.py`: package export for tests and agent tooling.

The runner does not re-implement the experiment logic. It delegates to the benchmark and
autoresearch scripts already in `scripts/`.
That keeps the catalog clean while avoiding duplicate experiment code.

## Suite Roster

The source of truth is `catalog.py`. Use `python numerical_experiments/run.py --list`
for full details, including scripts, heuristics, learned-policy families, and notes.

Current ready suites:

- `lost_sales_single_instance_check`
- `lost_sales_full_policy_grid`
- `fixed_cost_known_optimum_validation`
- `fixed_cost_single_instance_check`
- `fixed_cost_full_policy_grid`
- `dual_sourcing_reference_grid`
- `owmr_kaynov_full_paper_benchmark`

Current exploratory suites:

- `dual_sourcing_backbone_screen`
- `dual_sourcing_tree_autoresearch`
- `multi_echelon_autoresearch`
- `multi_echelon_gijs_full_paper_benchmark`

Ready suites are expected to be runnable entry points for benchmark or
paper-supporting workflows. Exploratory suites are still useful, but they are
policy-design or protocol-screening work and should be launched explicitly.

## Linux Usage

Build the environment and native extension on the Linux machine first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m pip install maturin
python scripts/rust/build_extension.py
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
