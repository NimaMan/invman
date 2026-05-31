# Experiments

This folder is the paper-facing benchmark home for `joint_replenishment`.

Planned use:

- define reported multi-item instances
- compare CMA-ES-optimized learned policies against the carried heuristics
- include the reduced exact DP comparator when tractable

Current code anchors:

- heuristics in `heuristics/`
- exact reduced benchmark in `finite_horizon_dp.rs`
- runnable benchmark script: `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`

Benchmark currently runnable WITHOUT a Rust rebuild (numbers reproduced 2026-05):

- literature anchor (setting 5, infinite-horizon value iteration): env optimal action at `(5,0)` is
  `q=(0,6)`, matching Vanvuchelen et al. (2020) Figure 3.
- repo reduced finite-horizon DP comparator (setting-1 family, 4 periods, discounted): optimal
  `(6,6)` cost `266.39`; MOQ `(7,5)` cost `386.10`; DYN-OUT `(6,6)` cost `383.96`.
- heuristic Monte-Carlo sweep over all 16 Table-2 settings (200 periods, 256 reps, discounted): MOQ
  is at or below DYN-OUT on every setting.

Not yet run (blocker): the learned soft-tree comparison. The rollout binding and the CMA-ES trainer
(`invman.es_mp.train` + `invman.policy.Policy`) are importable without a rebuild, but the helper
`scripts/joint_replenishment/common.py` imports a stale `invman.policies.soft_tree` path; a training
driver that uses the current `Policy` API is the remaining wiring.
