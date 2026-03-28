# Inventory Management Experiments

This repository now keeps one active code path:

- `invman.problems.lost_sales` for the canonical vanilla lost-sales package
- `invman.problems.dual_sourcing` for the Gijsbrechts / Veeraraghavan-Scheller-Wolf dual-sourcing settings
- `invman.problems.multi_echelon` for the Van Roy / Gijsbrechts two-echelon warehouse-retailer settings
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

Fresh post-refactor refresh runs on the same benchmark:

- heuristic validator still matches the trusted reference numbers: Myopic-1 `5.0641`, Myopic-2
  `4.8204`, SVBS `5.8349`
- fresh Rust-backed soft-tree rerun: `4.7658`
- fresh linear rerun: `5.0049`
- fresh NN `8x8` smoke rerun: `5.2504`

Interpretation:

- the Rust-backed soft-tree path remains healthy after the native/runtime refactor
- the linear and NN backbones are still more seed- and budget-sensitive, so the locked historical
  references above remain the stronger canonical baselines for those families

The current best learned architecture is:

- soft tree
- oblique splits
- depth `2`
- linear leaf outputs

This is better than `Myopic-2 = 4.8204` and is close to the known optimal reference `4.73`.

Trusted fixed-order-cost benchmark:

- lost sales with `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- demand `~ Poisson(5)`
- holding cost `h=1`

On this benchmark, the current best learned policy from the fixed-cost autoresearch loop is:

- soft tree
- oblique splits
- depth `1`
- linear leaf outputs
- `50k` eval: `8.77528`
- `1M` eval: `8.76576`

Current policy function approximator anchors on the same canonical instance:

- linear categorical quantity: `10.42369`
- NN gated ordinal quantity: `9.51636`
- transferred depth-2 soft tree: `8.81009`
- autoresearch-refined depth-1 soft tree: `8.76576`

Reference comparisons:

- earlier transferred depth-2 tree, `1M` eval: `8.81009`
- best heuristic on `1M` eval, modified `s,S,q`: `9.16537`

So the current best fixed-cost tree improves on the earlier tree by about `0.5%` and improves on
the best heuristic by about `4.36%` on the canonical fixed-cost instance.

Dual-sourcing smoke benchmark on the hardest small-scale literature instance `lr=4`, `ce=110`:

- learned oblique depth-2 soft tree with linear leaves: `249.84`
- best benchmark heuristic in the repo baseline: capped dual-index at `220.73`
- current interpretation: the dual-sourcing package is implemented and benchmarked, but tree policies are not yet competitive there under the first smoke budget

Dual-sourcing full-budget baseline on the same primary instance:

- learned oblique depth-2 soft tree with linear leaves: `233.08375`
- single-index: `226.816875`
- dual-index: `222.4025`
- capped dual-index: `221.61`
- tailored base-surge: `222.7825`

Current interpretation:

- the dual-sourcing training path is correct and reproducible;
- more CMA-ES budget helped materially versus the smoke run;
- but the current direct vector-action tree remains about `5.2%` worse than the best heuristic.

The current working hypothesis is that dual sourcing needs a better action representation:

- benchmark heuristics act on expedited and regular inventory positions, not directly on raw
  `(q_regular, q_expedited)` quantities;
- our current tree must learn both the inventory-position transform and the replenishment logic in one
  vector output space;
- the next likely-better family is a state-dependent target-position policy, for example a learned tree
  that outputs target expedited and regular positions and then maps those deterministically to orders.

Multi-echelon smoke benchmark on the larger Van Roy / Gijsbrechts setting:

- learned oblique depth-2 soft tree with linear leaves: `3776.45`
- best constant base-stock benchmark on the same evaluation: `3776.45`
- current interpretation: the first tree smoke run matched the best constant base-stock benchmark on the setting-2 action grid

## Quick Start

Create an environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m pip install maturin
```

Build the optional Rust extension into the active virtualenv:

```bash
python scripts/build_rust_extension.py
```

For agent-driven runs on another machine, use the repo-local guide in `AGENTS.md`.

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
- `invman/problems/lost_sales/`: vanilla lost-sales env, heuristics, and benchmark references
- `invman/problems/lost_sales_fixed_order_cost/`: fixed-order-cost extension and heuristic search
- `invman/problems/dual_sourcing/`: dual-sourcing env, heuristics, bounded DP, and literature settings
- `invman/problems/multi_echelon/`: two-echelon env, constant base-stock benchmark, and literature settings
- `invman/policies/`: canonical linear, neural, and tree policy parameterizations
- `rust/`: native rollout kernels used by the Rust-backed policy path
- `invman/es.py`, `invman/es_mp.py`: evolution-strategy optimizers and training loop
- `scripts/run_experiment.py`: single entry point for training and evaluation
- `numerical_experiments/`: curated experiment catalog and launcher for Linux-scale benchmark runs
- `scripts/lost_sales/autoresearch_tree_structures.py`: vanilla lost-sales tree-architecture comparison runner
- `scripts/lost_sales_fixed_order_cost/autoresearch_fixed_order_cost.py`: fixed-cost autoresearch runner
- `scripts/lost_sales_fixed_order_cost/autoresearch_fixed_order_tree_structures.py`: fixed-cost tree-architecture screening runner
- `scripts/dual_sourcing/autoresearch_dual_sourcing.py`: dual-sourcing autoresearch runner
- `scripts/multi_echelon/autoresearch_multi_echelon.py`: multi-echelon autoresearch runner
- `autoresearch/`: autoresearch-style loop docs for vanilla and fixed-cost benchmarks
- `../docs/benchmarks/lost_sales_l4_refresh.md`: refreshed vanilla lost-sales benchmark note after
  the Rust refactor
- `../docs/benchmarks/fixed_cost_l4_refresh.md`: canonical fixed-cost benchmark note

## Fixed Ordering Cost Variant

The environment already supports an optional `fixed_order_cost` parameter. That gives a clean extension path toward the lost-sales problem with a setup cost on positive orders. The literature note for that variant is tracked in `../docs/lost_sales_fixed_order_cost_literature.md`.

The fixed-order-cost benchmark layer and heuristic baselines are in place, and the current best
autoresearch-refined oblique depth-1 soft tree with linear leaves outperforms the benchmark
heuristic policies on the canonical fixed-cost instance.
