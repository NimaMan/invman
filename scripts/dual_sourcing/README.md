# scripts/dual_sourcing

Runnable scripts for the dual-sourcing problem (Gijsbrechts et al. 2022, Figure-9
family: regular lead time `l_r` in {2,3,4} x expedited cost `c_e` in {105,110}, six
reference rows). The environment, heuristic searches (single / dual / capped-dual
index, tailored base-surge) and the bounded average-cost DP all live in Rust and are
reached through the `invman_rust` extension; training goes through the flat `invman`
package (`invman.experiment_runner.run_experiment`, CMA-ES). Dual sourcing is
soft-tree-ONLY on the policy side.

The trained-policy-search artifacts and the warm-start training/eval drivers they
reuse live in `outputs/dual_sourcing_policy_search/` (see its README).

## Shared glue

- **`dual_sourcing_benchmark_lib.py`** — single source of shared orchestration:
  `build_reference_args` (Rust reference instance -> experiment args),
  `get_benchmark_grid`/`build_grid_instances`, `evaluate_default_heuristics` (Rust
  grid searches on a fixed demand path), `bounded_dp_optimal` (opt-in),
  `EXPERIMENT_SPECS` (soft-tree policy roster; the selected spec is the axis-aligned
  constant-leaf small-cap capped-dual-index tree), CMA-ES budgets
  (`screening`/`full`), `configure_run_args`, `result_path_for`, `learned_cost_of`.

## Seed-robust reporting (optimizer-seed depth, srp standard)

- **`seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py`** — THE seed-robust
  headline runner and the producer of the artifact that
  `paper/generate_results_tables.py`'s fail-loud gate reads
  (`outputs/dual_sourcing_policy_search/seed_robust_report.json`). For each of the
  six Fig-9 rows and each of >= 5 optimizer seeds (canonical 9001..9005, enforced via
  `invman.optimizer_seed_robustness_policy.run_over_seeds`), it retrains the chosen
  warm-start-at-CDI depth-2 axis/constant small-cap capped-dual-index soft tree
  (exactly the `train_warmstart.py` protocol, sigma_init 0.5, budget `full`) and
  evaluates it on the paired-CRN protocol against CDI (80 shared seeds x horizon
  100000, `eval_artifacts_highprec`). Per-instance blocks carry the standardized srp
  summary; top-level `n_optimizer_seeds` = MIN over instances. `--smoke` runs the
  same pipeline at a tiny budget and writes ONLY to
  `outputs/dual_sourcing_policy_search/smoke_seed_robust/` (never the real artifact).
  Usage (CPU-capped):
  `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/dual_sourcing/seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2`
- **`seed_robust_learned_vs_cdi_tier_c.py`** — seed-robust learned-vs-CDI on the ONE
  hardest *reachable* taxonomy cell (`dual_l2_ce110_b50_u08_catC`), suite eval
  protocol (not paired CRN). Complementary to the runner above; expected verdict
  there is parity (CDI's gap to the DP optimum is below sampling noise).
- **`aggregate_seed_robust_cdi.py`** — aggregates per-seed `benchmark_full_suite`
  instance JSONs (run under seed-distinct `--tags`) into a mean +/- std gap-vs-CDI
  table per row. Predates the srp module; the headline artifact now comes from
  `seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py`.

## Training / screening runners (single optimizer seed)

- **`autoresearch_dual_sourcing.py`** — train ONE soft-tree spec on one row; logs
  cost + gap vs the strongest heuristic (CDI).
- **`autoresearch_dual_sourcing_policy_structures.py`** — screen soft-tree STRUCTURE
  axes (depth, temperature, split, leaf, action adapter) on one row.
- **`autoresearch_dual_sourcing_backbones.py`** — compare soft-tree backbone choices
  (split geometry x leaf head) under a fixed budget.
- **`sweep_policy_variants.py`** — rank a roster of named soft-tree variants across
  rows by gap vs CDI.
- **`benchmark_full_suite.py`** — the full six-row suite: heuristics + (opt-in) DP +
  learned policies, per-instance comparative summaries.

## Verification / analysis

- **`validate_reference_grid.py`** — reproduce the published Figure-9 heuristic
  ranking/gaps from the Rust backend (env-faithfulness check).
- **`cdi_gap_to_optimum_regime_sweep.py`** / **`cdi_out_of_sample_gap_to_optimum.py`**
  — CDI's gap to the bounded-DP optimum across demand/cost regimes, in-sample and
  out-of-sample (basis of the instance taxonomy in
  `docs/benchmarks/DUAL_SOURCING_INSTANCE_TAXONOMY_2026_06_07/README.md`).
