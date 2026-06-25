# scripts/ameliorating_inventory — folder functionality

Code for the **ameliorating inventory** family (products that improve with age:
spirits, port wine). Two env generations exist in the Rust crate; the headline work
targets the **faithful average-PROFIT blending env**
(`src/problems/ameliorating_inventory/average_profit_blending_env.rs`, the
Pahr & Grunow 2025 model). This problem is **profit-maximizing** — watch sign
conventions everywhere.

## Files

### `autoresearch_ameliorating_inventory_average_profit.py`
The single-seed training entry point on the faithful average-profit env.
- Instances: `spirits_0001` (default), `spirits_0002`, `spirits_1002`, `port_wine`,
  each anchored to its published perfect-information LP **upper bound** (context, not
  a same-protocol comparator).
- Policy: depth-1 oblique soft tree, linear leaf, scalar price-reactive **purchase**
  head; CMA-ES warm-started at an order-up-to purchase encoding.
- Gate: best order-up-to level tuned on the held-out eval block (`tune_order_up_to`).
- Deployment: best-of candidate set {order-up-to anchor, CMA xbest, CMA xfavorite
  (`--deploy_endpoint floor` default), best per-generation incumbent} on the held-out
  paired-CRN eval block.
- Budgets: `smoke` / `screening` / `full` (see `BUDGETS`).
- Writes `outputs/autoresearch/<run_tag>/...json` + `results.tsv`.

### `seed_robust_ameliorating_inventory.py`
The **seed-robust headline runner** (optimizer-seed robustness standard,
`invman/optimizer_seed_robustness_policy.py`, problem_id `ameliorating_inventory`,
mode="seeds", >= 5 seeds). Imports the autoresearch module above via importlib and
reuses its helpers verbatim (no env/policy/Rust changes); re-runs the exact
single-seed protocol once per optimizer seed (default canonical 9001..9005, gate
re-tuned per seed on that seed's eval block, paired) and aggregates with
`srp.run_over_seeds` (sample std, shared verdict rule).
- **Sign convention**: srp assumes lower-is-better, so the per-seed records feed it
  NEGATED profits (`gate_cost = -gate_profit`, `best_learned_cost = -learned_profit`)
  and `savings_pct_vs_gate` is PRECOMPUTED profit-oriented
  (`100*(learned-gate)/|gate|`, positive = learned beats gate). Profit-oriented
  convenience keys (`learned_profit_seed_mean/std`, `gate_profit_seed_mean/std`) sit
  alongside the standardized srp keys in the output JSON.
- Real artifact: `outputs/ameliorating_inventory/seed_robust_report.json`
  (non-default instance → `seed_robust_report_<instance>.json`).
- `--smoke`: forces the tiny `smoke` budget preset + `mp_num_processors 1` and writes
  ONLY to `outputs/ameliorating_inventory/smoke_seed_robust/` — a smoke run can never
  clobber the real artifact.
- Full-budget usage (CPU-capped):
  `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/ameliorating_inventory/seed_robust_ameliorating_inventory.py --instance spirits_0001 --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2`

### `benchmark_repo_native_instance.py`
Internal-only benchmark of installed-binding policies on the repo-native
PRIMARY_REFERENCE_INSTANCE of the older finite-horizon discounted-cost env.
Explicitly NOT literature-comparable (see its honest-scope docstring); documents the
binding blockers for exact-optimal comparison.

### `RESULTS_FULL_BUDGET/README.md`
Embedded full-budget single-seed numbers (outputs/ is gitignored). Single-seed —
superseded as a headline by the seed-robust runner above.

### `PAPER_SECTION_DRAFT/README.md`
Draft paper subsection text for this family (editorial pass required before use).
