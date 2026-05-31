# one_warehouse_multi_retailer

Rust-first problem home for `one_warehouse_multi_retailer`, modeled after
Kaynov et al. (2024), IJPE 267, 109088.

## Formulation

- one upstream warehouse, `K` downstream retailers (divergent two-echelon)
- per-period decisions: warehouse order, retailer orders, and downstream **allocation** of
  scarce warehouse stock to retailers
- customer behavior is one of `lost_sales`, `backorder` (complete), or `partial_backorder`
  (the paper's three regimes)

Order of events implemented in `env.rs::step_state` (verified by the worked-transition test):

1. the warehouse pipeline head arrives; available warehouse = on-hand + arrival
2. retailer shipments leave warehouse stock (must not exceed available warehouse inventory)
3. the warehouse order enters the tail of the warehouse pipeline; each retailer shipment
   enters the tail of that retailer's pipeline
4. each retailer's pipeline head arrives; demand is realized against (retailer on-hand +
   arrival); under `partial_backorder` an emergency shipment may be drawn from remaining
   warehouse stock with probability `emergency_shipment_probability`
5. costs: warehouse holding on ending warehouse on-hand; per-retailer holding on ending
   on-hand; penalty on unmet (lost / backordered) units

Cost convention: the env reports a **positive period cost** and `reward = -period_cost`.
Published Kaynov rows are stored as negative reward; the script layer compares against
`-published_reward`.

## Verification status: `partial`

Two distinct claims (kept separate, as in `multi_echelon/divergent_special_delivery`):

- **Env transition + cost: faithful and exact-DP-validated.** The worked-transition test
  traces a full period by hand; the reduced finite-horizon DP confirms the heuristics are
  dominated by the true optimum (`optimal 8.485 <= proportional/min_shortage 9.2225`,
  reproduced live via `one_warehouse_multi_retailer_exact_dp_summary()`).
- **Published numbers: approximately reproduced, not bit-matched.** Repo echelon base-stock +
  allocation heuristics land ~1-6% off the published Kaynov rows, with a regime-dependent sign
  (lost-sales within ~1-2.5% below; backorder ~3.6-5.5% below; partial-backorder ~6% above).
  This is a protocol / initial-condition residual (mean-filled warm start + repo search grid),
  not a transition bug. `VERIFICATION_PROBLEM_INSTANCE` carries `literature_verified = false`.

Full cost-row table, corroboration of the carried PPO bands, and root-cause discussion are in
`literature/README.md`.

## Benchmark

- **Runnable now (heuristics vs published, no Rust rebuild):**
  `scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py` — self-contained,
  imports only `invman_rust`. Grid-searches echelon base-stock levels for proportional and
  min-shortage allocation, evaluates at 1000 trajectories, and prints repo-vs-published gaps
  plus the exact-DP self-consistency check.
- **Blocked (learned soft-tree vs heuristics):** the older `run_paper_benchmark.py` / `common.py`
  import `invman.policies.soft_tree.SoftTreePolicy`, which no longer exists after the repo's
  policy refactor. The cached soft-tree numbers in `experiments/reports/latest_report.json` thus
  cannot be regenerated against the current install without repointing those scripts at the
  refactored policy builder. The Rust rollout bindings are present and working; only the Python
  policy wrapper import is stale. See `next_steps` in `literature/README.md`.

## Code layout

- `env.rs` — raw state + `step_state` transition/cost (raw state quantities only; no
  normalization in the environment layer)
- `allocation.rs` — `proportional`, `random_sequential`, `min_shortage` rules
- `demand.rs` — Poisson / rounded-normal / discrete-uniform / deterministic demand
- `heuristics/echelon_base_stock.rs` — echelon base-stock order computation
- `finite_horizon_dp.rs` — reduced exact DP + heuristic / soft-tree evaluators
- `rollout.rs` — soft-tree rollout, feature construction, action modes
  (`direct_orders` / `echelon_targets` / `symmetric_echelon_targets`)
- `references.rs` — `KAYNOV_2024_REFERENCE`, 14-instance `TABLE_A3_INSTANCES`,
  `PRIMARY_REFERENCE_INSTANCE` (= instance 7), `WORKED_TRANSITION_REFERENCE`,
  `VERIFICATION_PROBLEM_INSTANCE`
- `bindings.rs` — pyo3 bindings
- anchors live under `literature/`, `verification/`, `experiments/`, `practical/`

## Benchmark notes

- proportional rationing must exhaust available warehouse inventory; floor-only rounding is not
  a valid benchmark implementation (asserted in tests)
- the Kaynov reproduction uses a mean-filled pipeline warm start in the script layer rather than
  an empty-system cold start; this is a repo evaluation choice, not a published protocol, and is
  the leading suspect for the residual reproduction gap
- learned policies train with `random_sequential` allocation and evaluate with `proportional`,
  matching Kaynov's training-allocation idea
- for symmetric retailer instances, the preferred learned action space is
  `symmetric_echelon_targets`: one warehouse target and one shared retailer target, expanded into
  retailer orders inside the rollout layer
- `literature_verified` is the only verification-status label carried by references; it labels the
  carried instance/row provenance, not a tight numerical reproduction
