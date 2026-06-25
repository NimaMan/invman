# scripts/perishable_inventory

Runnable scripts for the single-item, periodic-review **perishable inventory** problem
(De Moor, Gijsbrechts & Boute 2022 / Farrington, Wong, Li & Utley 2025 Scenario A),
implemented in Rust and exposed to Python through the `invman_rust` extension
(`perishable_inventory_*` bindings: reference catalog, exact-MDP value iteration,
base-stock / BSP-low-EW search, soft-tree population rollouts; all returns are
**discounted negated costs**, gamma 0.99, higher / less negative is better).

The reference catalog (`perishable_inventory_list_reference_instances`) carries the 32
De Moor Scenario-A settings (`de_moor2022_m{2..5}_exp{1..8}_*`); the primary anchor is
`de_moor2022_m2_exp2_l1_cp7_fifo` (m=2, L=1, FIFO, best base-stock S=7, VI optimum
-1457), whose exact MDP the crate re-derives against the published tables.

## Scripts

- **`autoresearch_perishable_inventory.py`** — the SINGLE-SEED learned-policy runner
  behind the paper rows: warm-starts CMA-ES at the encoded best base-stock (generation 0
  reproduces the gate), trains a depth-2 oblique-linear soft tree under paired CRN,
  selects the promoted candidate on a **disjoint validation block** (the load-bearing
  fix), and reports the held-out eval return vs the best base-stock GATE (the
  apples-to-apples comparator; the analytic VI optimum is context only). Budgets:
  `smoke` / `screening` / `full`. Writes a JSON payload + a TSV ledger row under
  `outputs/autoresearch/<run_tag>/`.

- **`seed_robust_perishable_inventory.py`** — the SEED-ROBUST aggregator for the paper
  headline (optimizer-seed robustness standard,
  `invman/optimizer_seed_robustness_policy.py`, problem_id `perishable_inventory`,
  mode=seeds, canonical seeds 9001..9005). Reuses `autoresearch_perishable_inventory.py`
  **verbatim** (importlib + patched argv, one `main()` call per instance x seed), maps
  returns to cost-style records (`gate_cost` / `best_learned_cost` = negated returns),
  and aggregates with `srp.run_over_seeds` into per-instance blocks carrying the
  standardized summary keys (`learned/gate_seed_mean/std`, `savings_pct_seed_mean/std`,
  `frac_seeds_beating_gate`, `verdict_vs_same_protocol_gate`). Default instances = the
  paper's two exact-MDP anchors (m2 FIFO primary + m2 LIFO).
  Real artifact: `outputs/perishable_inventory/seed_robust_report.json`.
  `--smoke` forces the tiny `smoke` budget preset, 1 CPU worker, and a **separate**
  artifact (`outputs/perishable_inventory/smoke_seed_robust/seed_robust_report_smoke.json`);
  a smoke run can never write the real report path.
  Full-budget usage:
  `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/perishable_inventory/seed_robust_perishable_inventory.py --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2`

- **`run_exact_slice_benchmark.py`** — self-contained exact-slice benchmark on the
  m=2 / L=1 settings: exact tabular optimum (value iteration) vs best base-stock vs best
  BSP-low-EW vs CMA-ES soft trees (linear and sigmoid_linear leaves), all on the same
  discounted-return protocol.

- **`run_paper_benchmark.py`** — paper-instance benchmark runner over the De Moor
  reference settings (heuristic searches + soft-tree training via `common.py`),
  writing per-instance artifacts under `outputs/perishable_inventory/`.

- **`run_practical_benchmark.py`** — scores policies on the practical
  grocery-like daily demand trace via `invman.benchmarks.practical`.

- **`train_soft_tree_reference.py`** — minimal soft-tree CMA-ES trainer for one
  reference instance (the early single-run entry point superseded by
  `autoresearch_perishable_inventory.py` for paper results).

- **`validate_against_papers.py`** — checks the env against the published De Moor /
  Farrington numbers (exact-MDP summaries, base-stock levels, Table-3 returns).

- **`common.py`** — shared helpers for the older benchmark scripts (reference loading,
  zero start state, policy evaluation wrappers). The autoresearch and seed-robust
  runners are self-contained and do not use it.

## Verification (Rust)

Literature verification lives in the crate, not here:
`invman_rust.perishable_inventory_exact_mdp_summary(name)` re-derives the published VI
mean returns, best base-stock levels, and optimal-policy tables at call time
(`matches_published_*` flags), and `cargo test --release perishable` exercises the env.
