# lost-sales autoresearch

This is an adaptation of Andrej Karpathy's `autoresearch` setup to the inventory-management repo.

## Setup

To set up a new run:

1. Agree on a `run_tag`.
2. Read these files for context:
   - `README.md`
   - `autoresearch/README.md`
   - `autoresearch/program_lost_sales.md`
   - `scripts/lost_sales/benchmark_full_suite.py`
   - `src/problems/lost_sales/vanilla/reference_costs.rs`
   - `invman/policy.py`
   - `invman/policy_registry.py`
   - `invman/rollout_fitness.py`
   - `src/problems/lost_sales/vanilla/rollout.rs`
3. Rebuild the Rust extension if any Rust files changed:
   - `python scripts/build_rust_extension.py`
4. Verify the baseline code path works:
   - `python scripts/lost_sales/validate_reference_instance.py --num_seeds 3`

## Scope

The benchmark is fixed to the trusted vanilla lost-sales instance:

- `L=4`
- `p=4`
- demand `~ Poisson(5)`
- `h=1`

The evaluation harness is fixed. Do not modify:

- `scripts/lost_sales/benchmark_full_suite.py`
- `src/problems/lost_sales/vanilla/reference_costs.rs`
- `scripts/lost_sales/validate_reference_instance.py`
- heuristic implementations used as the benchmark baseline

The intended search surface is:

- `invman/policy.py`
- `invman/policy_registry.py`
- `invman/policy_build.py`
- `invman/rollout_fitness.py`
- Rust policy math under `src/core/policies/`
- limited support code needed to wire policy evaluation into the existing training loop

## Experiment budgets

Use the fixed budgets from `scripts/lost_sales/autoresearch_lost_sales.py`:

- `screening`: fast search budget
- `full`: trusted benchmark budget

The first run in any new `run_tag` should establish a baseline.
Promising ideas may then be promoted beyond the screening budget. The important requirement is that
the benchmark and evaluation protocol stay fixed.

## Experiment loop

For each experiment:

1. Make one policy-focused change.
2. Run:
   - `python scripts/lost_sales/autoresearch_lost_sales.py --run_tag <tag> --budget screening --description "<what changed>" ...`
3. The script writes:
   - experiment JSON under `outputs/autoresearch/<tag>/results/`
   - logs under `outputs/autoresearch/<tag>/logs/`
   - models under `outputs/autoresearch/<tag>/models/`
   - a TSV ledger at `outputs/autoresearch/<tag>/results.tsv`
4. Keep changes only if the learned policy improves the best kept learned cost so far and the code remains reasonably simple.
5. Promote promising ideas to `--budget full`.

## Goal

Lower the learned policy cost on the fixed lost-sales benchmark while preserving a clean, general policy-learning pipeline.

Current best architecture:

- oblique soft tree
- depth `2`
- linear leaf outputs
- benchmark cost `4.753725`
