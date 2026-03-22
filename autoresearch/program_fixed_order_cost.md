# fixed-order-cost autoresearch

This is the fixed-order-cost counterpart to `program_lost_sales.md`.

## Setup

To set up a new run:

1. Agree on a `run_tag`.
2. Read these files for context:
   - `README.md`
   - `autoresearch/README.md`
   - `autoresearch/program_fixed_order_cost.md`
   - `invman/problems/lost_sales_fixed_order_cost/reference_instances.py`
   - `invman/problems/lost_sales_fixed_order_cost/heuristics.py`
   - `invman/policies/`
   - `rust/src/policies/`
   - `rust/src/rollout/`
3. Rebuild the Rust extension if any Rust files changed:
   - `python scripts/build_rust_extension.py`
4. Verify the fixed-cost benchmark code path works:
   - `python scripts/validate_fixed_order_cost_heuristics.py --reference_instance lit_pois_mu5_l4_p4_k5`

## Scope

The benchmark is fixed to the canonical fixed-order-cost instance:

- `L=4`
- `p=4`
- `K=5`
- demand `~ Poisson(5)`
- `h=1`

The evaluation harness is fixed. Do not modify:

- `invman/problems/lost_sales_fixed_order_cost/reference_instances.py`
- heuristic search code used as the benchmark baseline
- the long-run evaluation protocol when checking promoted candidates

The intended search surface is:

- `invman/policies/`
- `rust/src/policies/`
- `rust/src/rollout/`
- limited support code needed to wire policy evaluation into the training loop

## Experiment budgets

Use the fixed budgets from `scripts/autoresearch_fixed_order_cost.py`:

- `screening`: fast search budget
- `full`: trusted benchmark budget

The first runs in any new `run_tag` should re-establish the current tree baseline before trying
new policy structure changes. Promising ideas may then be promoted beyond the screening budget.

## Experiment loop

For each experiment:

1. Make one policy-focused change.
2. Run:
   - `python scripts/autoresearch_fixed_order_cost.py --run_tag <tag> --budget screening --description "<what changed>" ...`
3. The script writes:
   - experiment JSON under `outputs/autoresearch/<tag>/results/`
   - logs under `outputs/autoresearch/<tag>/logs/`
   - models under `outputs/autoresearch/<tag>/models/`
   - a TSV ledger at `outputs/autoresearch/<tag>/results.tsv`
4. Keep changes only if the learned policy improves the best kept learned cost so far and the code remains reasonably simple.
5. Promote promising ideas to `--budget full`.
6. Re-evaluate the promoted winner on a long horizon:
   - `python scripts/evaluate_saved_policy.py --problem lost_sales_fixed_order_cost --reference lit_pois_mu5_l4_p4_k5 --model_dir <model_dir> --eval_horizon 1000000 --eval_seeds 3`

## Goal

Lower the learned policy cost on the fixed-order-cost benchmark while preserving a clean, general
policy-learning pipeline.

Current best architecture:

- oblique soft tree
- depth `1`
- linear leaf outputs
- benchmark cost `8.76576` on the 1M evaluation
