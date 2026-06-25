# `invman/` package — functionality map

One entry per module; each module also carries a full algorithmic description in
its top-of-file docstring.

## Optimization & training
- **`cmaes.py`** — thin wrapper around `cma.CMAEvolutionStrategy`; param-scale-
  normalized search space; optional warm start (`x0`) used to seed the CMA mean at
  an encoded heuristic (e.g. CDI for dual sourcing).
- **`es_mp.py`** — the multiprocess CMA-ES training loop (`train`): ask/eval/tell
  over the rollout oracle, population fitness via worker pool.
- **`ppo_trainer.py`** — PPO baseline trainer (cross-protocol context only; never
  a same-protocol comparator).
- **`experiment_runner.py`** — single-optimizer-seed experiment driver:
  `run_experiment(args)` = seed → build policy → CMA-ES train → evaluate over
  `eval_seeds` *evaluation (demand-path) seeds* → write result JSON.
  `summarize_costs` reports `num_evaluation_seeds` (with the legacy `num_seeds`
  alias) — evaluation seeds, NOT optimizer seeds.

## Robustness standard (single source of truth)
- **`optimizer_seed_robustness_policy.py`** — declares per problem how headline
  robustness to CMA-ES optimizer randomness is established:
  *breadth* (lost_sales, lost_sales_fixed_order_cost: one optimizer run per cell
  across the full 24/48-instance grid) vs *seeds* (every other problem: ≥5
  independent optimizer seeds, sample n−1 std, mean±std primary). Provides the
  shared aggregation (`summarize_values`, `build_seed_robust_summary`), the
  shared verdict rule (ROBUST_BEAT / BEAT_WITHIN_STD / PARITY / ROBUST_LOSS), the
  generic `run_over_seeds` driver used by all `scripts/*/seed_robust_*.py`
  runners, and fail-loud guards (`assert_seed_policy`, `assert_breadth_grid`)
  consumed by `paper/generate_results_tables.py`. Fail-closed: unregistered
  problems are held to seeds/≥5.

## Policies
- **`policy.py` / `policy_build.py` / `policy_common.py`** — policy function
  approximators (linear, NN, oblique soft decision trees) and construction.
- **`policy_registry.py`** — named policy specs (backbone, decoder/action-output
  mode, tree shape) and `args` wiring.
- **`dual_sourcing_policy_spec.py`** — dual-sourcing-specific policy specs
  (capped-dual-index action geometries).

## Evaluation & infra
- **`rollout_fitness.py`** — fitness via the Rust rollout oracle (`invman_rust`);
  model/population evaluation entry points.
- **`lost_sales_reference.py`** — heuristic baselines from the Rust reference-cost
  catalog.
- **`config.py`** — the shared argparse surface (`--problem`, `--seed` = base
  optimizer seed, `--eval_seeds` = evaluation seeds, budgets, env knobs).
- **`cpu_limits.py`** — process CPU caps (`RAYON_NUM_THREADS`/`OMP` pinning,
  `mp_num_processors` normalization); keep total ~4 cores for benchmark suites.
- **`es_mp.py` / `utils.py`** — run-status tracking, global seeding, misc.
