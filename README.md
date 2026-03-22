# Inventory Management Experiments

This repository now keeps one active code path:

- `invman.env.lost_sales.LostSalesEnv` for the periodic-review lost-sales problem
- policy optimization with evolution strategies over compact policy parameterizations
- a canonical `invman.policies` package for learned policy classes
- a colocated Rust crate under `rust/` for high-throughput rollout kernels
- one runner script at `scripts/run_experiment.py`

The current baseline problem is the single-item lost-sales setting with lead time, holding cost, shortage cost, and either Poisson or geometric demand. The runner can train either a linear policy or a small neural policy and compare the learned policy against the classic lost-sales heuristics already in the repo.

## Current Findings

Trusted vanilla benchmark:

- lost sales with `L=4`
- shortage cost `p=4`
- demand `~ Poisson(5)`
- holding cost `h=1`

Current learned-policy reference points on that benchmark:

- linear policy: `4.8066`
- earlier soft-tree benchmark: `4.7980`
- current best learned policy: `4.753725`

The current best learned architecture is:

- soft tree
- oblique splits
- depth `2`
- linear leaf outputs

This is better than `Myopic-2 = 4.8204` and is close to the known optimal reference `4.73`.

Trusted fixed-order-cost transfer benchmark:

- lost sales with `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- demand `~ Poisson(5)`
- holding cost `h=1`

On this benchmark, the same soft-tree family transfers strongly:

- learned soft tree, `50k` eval: `8.81895`
- learned soft tree, `1M` eval: `8.81009`
- best heuristic on `1M` eval, modified `s,S,q`: `9.16537`

So the current tree policy is about `3.9%` better than the best heuristic on the canonical
fixed-cost instance.

## Quick Start

Create an environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Build the optional Rust extension into the shared virtualenv:

```bash
python scripts/build_rust_extension.py
```

Run a small experiment:

```bash
python3 scripts/run_experiment.py --training_episodes 20 --horizon 200 --eval_horizon 2000 --eval_seeds 5
```

Outputs are written under `outputs/`:

- `outputs/logs/`
- `outputs/models/`
- `outputs/results/`

## Structure

- `invman/config.py`: CLI configuration
- `invman/env/lost_sales.py`: environment and rollout helpers
- `invman/heuristics/lost_sales_heuristics.py`: Myopic-1, Myopic-2, SVBS
- `invman/policies/`: canonical linear, neural, and tree policy parameterizations
- `rust/`: native rollout kernels used by the Rust-backed policy path
- `invman/es.py`, `invman/es_mp.py`: evolution-strategy optimizers and training loop
- `scripts/run_experiment.py`: single entry point for training and evaluation
- `scripts/autoresearch_tree_structures.py`: tree-architecture comparison runner
- `autoresearch/`: autoresearch-style loop for the trusted lost-sales benchmark

## Fixed Ordering Cost Variant

The environment already supports an optional `fixed_order_cost` parameter. That gives a clean extension path toward the lost-sales problem with a setup cost on positive orders. The literature note for that variant is tracked in `../docs/lost_sales_fixed_order_cost_literature.md`.

The fixed-order-cost benchmark layer and heuristic baselines are in place, and the current best
oblique depth-2 soft tree with linear leaves now also outperforms the benchmark heuristic policies
on the canonical fixed-cost instance.
