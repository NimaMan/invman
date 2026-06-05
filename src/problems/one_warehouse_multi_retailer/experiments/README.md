# Experiments

This folder is the paper-facing benchmark home for `one_warehouse_multi_retailer`.

Current paper suite:

- reported instances: the 14 Kaynov Table A.3 instances carried in `references.rs`
- learned policy family: CMA-ES soft trees
- heuristic baselines: echelon base-stock with proportional allocation and min-shortage allocation
- published learned-policy comparator: PPO row carried from Kaynov Table A.3

Kaynov protocol used by the suite:

- heuristic search uses 1000 trajectories of length 100 with common random numbers
- benchmark evaluation uses 1000 trajectories of length 100
- learned policies are evaluated with proportional allocation
- the default learned-policy training protocol in the repo follows the paper's training-allocation idea:
  train with `random_sequential`, evaluate with `proportional`

Important inference:

- for instance 14, Kaynov state that the heuristic benchmark searches over warehouse level `z0` and a shared percentile parameter `k` for retailer targets, but the paper does not publish a discrete `k` grid
- the repo therefore enumerates the unique integer retailer-target vectors induced by continuous `k in [0, 3]`

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
- paper runner in `scripts/one_warehouse_multi_retailer/run_paper_benchmark.py`

Default outputs:

- `src/problems/one_warehouse_multi_retailer/experiments/reports/latest_report.json`
- `src/problems/one_warehouse_multi_retailer/experiments/reports/README.md`
