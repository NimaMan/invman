# one_warehouse_multi_retailer

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "one_warehouse_multi_retailer",
  "instance_id": "kaynov2024_instance_7",
  "instance_parameters": {
    "replications": 1000,
    "seed": 2222
  },
  "policy": "lost_sales_echelon_base_stock_min_shortage",
  "metric": "simulation_cost",
  "expected_value": 1408.08,
  "reference": {
    "citation": "Kaynov et al. (2024) published one-warehouse multi-retailer benchmark row",
    "locator": "instance 7 min-shortage benchmark cost, standard error 0.95",
    "doi_or_url": "https://doi.org/10.1016/j.ijpe.2023.109088",
    "literature_verified": false,
    "notes": "A published heuristic simulation row that reproduces within a loose stochastic tolerance; not a strict literature optimum."
  },
  "code_value": 1394.8165,
  "tolerance": {
    "relative_percent": 1.2
  },
  "command": "python scripts/one_warehouse_multi_retailer/validate_reference_instance.py \\\n  --reference_name kaynov2024_instance_7 \\\n  --benchmark_replications 1000 \\\n  --seed 2222"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `published_heuristic_simulation_match` |
| Instance | `kaynov2024_instance_7` |
| Metric | lost-sales echelon-base-stock min-shortage benchmark cost |
| Literature value | `1408.08` cost, standard error `0.95` |
| Current repo value | `1394.8165` cost with `1000` replications and seed `2222` |
| Tolerance | `1.2%` relative gap for this stochastic reproduction |
| Last validated | `2026-06-22` |

### Source

Kaynov et al. (2024), International Journal of Production Economics 267, 109088, DOI `10.1016/j.ijpe.2023.109088`, Table 1 / Table A.3 as carried in `references.rs`.

### Validation command

```bash
python scripts/one_warehouse_multi_retailer/validate_reference_instance.py \
  --reference_name kaynov2024_instance_7 \
  --benchmark_replications 1000 \
  --seed 2222
```

Expected output includes:

```text
echelon_base_stock_min_shortage published 1408.080 repo 1394.816 relative gap -0.942%
```

### Notes

The sign convention differs between the paper-carried reward/profit values and this validation script's positive cost display. This file uses positive cost magnitudes. The full problem remains hard: only a subset of Kaynov rows reproduce tightly, and the exact DP anchor is repo-native rather than a full 100-period literature optimum.

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

- **Heuristics vs published (no Rust rebuild):**
  `scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py` — self-contained,
  imports only `invman_rust`. Grid-searches echelon base-stock levels for proportional and
  min-shortage allocation, evaluates at 1000 trajectories, and prints repo-vs-published gaps
  plus the exact-DP self-consistency check.
- **Learned soft-tree vs heuristic vs published (REGENERATED 2026-05-31):**
  `scripts/one_warehouse_multi_retailer/benchmark_learned_vs_heuristic.py` — trains a soft-tree
  with CMA-ES (`invman.es_mp.train` + the `..._soft_tree_population_rollout` binding +
  `invman.policy.Policy`, `symmetric_echelon_targets` action mode), then evaluates the learned
  weights and the grid-searched echelon base-stock heuristic on the SAME held-out demand paths
  (Common Random Numbers via the `*_from_paths` bindings) so the comparison is out-of-sample and
  paired. The `common.py` migration to `invman.policy.Policy` removed the previous blocker
  (`invman.policies.soft_tree.SoftTreePolicy` no longer needed).

  Representative subset (one per regime; full budget: depth 2, popsize 32, 600 CMA-ES
  generations, train_seed_batch 12; held-out 4096 paths; 100-period undiscounted cost):

  | Instance | CB | Learned | Best Heuristic | Published PPO | Learned vs Heuristic | Winner |
  | --- | --- | ---: | ---: | ---: | ---: | --- |
  | `kaynov2024_instance_1` | `backorder` | `1584.45` | `1558.12` | `1637.20` | `-1.69%` | heuristic |
  | `kaynov2024_instance_6` | `lost_sales` | `1370.50` | `1348.05` | `1347.34` | `-1.67%` | heuristic |
  | `kaynov2024_instance_11` | `partial_backorder` | `1189.51` | `1184.46` | `971.86` | `-0.43%` | heuristic |

  The tuned base-stock + allocation heuristic wins on all three, but only by `0.4%`–`1.7%` on
  the held-out CRN block; the learned soft-tree is competitive, not dominant — expected for
  these symmetric Poisson(3) instances where base-stock is near-optimal. Full per-allocation
  numbers, standard errors, and the budget/protocol are in
  `outputs/one_warehouse_multi_retailer/learned_benchmark/learned_vs_heuristic_results.json` and
  `experiments/reports/README.md`. The older depth-1 14-instance cache in
  `experiments/reports/latest_report.json` is kept for historical reference. Instances 2–5,
  7–10, 12–14 were not re-run in this pass (see coverage note in `experiments/reports/README.md`).

## Autoresearch

Because the learned soft-tree currently *loses* to the tuned heuristic by 0.4%-1.7% on the three
instances above, there is a single-policy autoresearch loop to search the policy/control surface and
try to flip the sign. The program file
`policy_search/programs/program_one_warehouse_multi_retailer.md` states the trusted benchmark (the Kaynov
instances + the grid-searched echelon base-stock heuristic under the better of
`{min_shortage, proportional}` allocation), the editable search surface (tree depth / temperature /
split `{oblique, axis_aligned}` / leaf `{constant, linear}`; action design
`{symmetric_echelon_targets, direct_orders}`; allocation policy; CMA-ES warm-start at the best
base-stock levels), the screening vs full budgets, and the keep/discard goal (beat the strongest
heuristic out-of-sample on a losing instance).

The runner `scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py` trains
ONE soft-tree with a CLI-selected structure on a named instance (REUSING `common.py` +
`benchmark_learned_vs_heuristic.py` helpers — same env, same heuristic grid search, same paired
held-out CRN evaluation), then appends a TSV ledger row (learned cost, best heuristic, gap, gap%,
structure flags) to `outputs/autoresearch/<run_tag>/results.tsv`. Run it capped at 2 cores:

```
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py \
    --description "warm-start at best base-stock, partial_backorder" \
    --budget screening --reference kaynov2024_instance_11 \
    --leaf_type constant --warm_start_at_best_base_stock
```

`--budget smoke` is an end-to-end validation preset only (popsize 8, 8 generations); use `screening`
to rank levers and `full` (popsize 32, 600 generations, 4096 held-out paths) to certify a flip.

### Autoresearch outcome (2026-05-31, full-budget sweep)

A focused warm-start-centric sweep (8 screening + 10 full-budget configs, CPU-capped at
`RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, `mp_num_processors=1`) closed the held-out gap to the
strongest heuristic to **exactly 0.0%** on all three losing instances — a tie, not a strict win.
Best config (per instance, all three): **depth-2 `axis_aligned` `constant` leaf, temperature 0.05,
`symmetric_echelon_targets`, CMA-ES warm-started at the best echelon base-stock (W, R)**:

| Instance | CB | Best learned (alloc) | Best heuristic (alloc) | gap% (full budget) | Prior gap% | Winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `kaynov2024_instance_1` | `backorder` | `1558.12` (min_shortage) | `1558.12` (min_shortage) | `0.0000%` (tie) | `-1.69%` | tie (heuristic-equal) |
| `kaynov2024_instance_6` | `lost_sales` | `1348.05` (proportional) | `1348.05` (proportional) | `0.0000%` (tie) | `-1.67%` | tie (heuristic-equal) |
| `kaynov2024_instance_11` | `partial_backorder` | `1184.46` (proportional) | `1184.46` (proportional) | `0.0000%` (tie) | `-0.43%` | tie (heuristic-equal) |

The learned cost equals the heuristic cost to six decimals: the warm-started constant-leaf tree
reproduces the heuristic action at generation 0 and CMA-ES (600 generations, sigma 1.5) finds **no
profitable state-dependent deviation**. This confirms the program's prior — on these symmetric
Poisson(3) K=3 instances the tuned echelon base-stock + allocation heuristic is at/near the optimum,
so there is no exploitable state structure for a learned policy to win on. No config produced a
robust strict flip (`learned < heuristic` beyond the held-out stderr `~1.4–2.4`).

Search coverage / what moved the needle:

- **Warm-start is decisive and was previously broken.** The original
  `_warm_start_flat_params` wrote the raw base-stock target into the leaf block, but the soft-tree
  passes leaf outputs through a per-leaf-type transform before grid-snapping
  (`src/core/policies/soft_tree.rs`): a `constant` leaf is `min + sigmoid(param)*(max-min)`,
  a `linear` leaf is `min + softplus(bias + w·state)`. Writing the raw target sigmoid-saturated the
  constant leaf to the grid max (gen-0 holdout ≈ 1879 vs heuristic ≈ 1180 on instance 11), so the
  warm-start started from a badly over-stocked policy, not the heuristic. The fix inverts the
  transform (logit for the constant leaf; zeroed leaf weights + softplus-inverse bias for the linear
  leaf) so gen-0 reproduces the heuristic exactly. After the fix, warm-started constant beats both
  the no-warm control (`-0.20%`/`-0.04%` full budget) and every linear/oblique/depth-3 variant.
- **Levers swept** (per instance, prioritized as the program flags): leaf `{constant, linear}`,
  depth `{2, 3}`, temperature `{0.05, 0.10, 0.20}`, split `{axis_aligned, oblique}`, warm-start
  `{on, off}`. Constant + axis-aligned + warm-start was best everywhere; `linear` leaves and
  `depth-3` only added parameters CMA-ES could not exploit (depth-3 ties at full budget but adds no
  value; linear is strictly worse, `-0.32%` to `-1.84%`). Temperature was immaterial under the
  warm-started constant leaf (gen-0 already at the heuristic).
- **Not run** (bounded sweep): `direct_orders` / `vector_quantity` action design (an expressiveness
  ablation; the symmetric geometry already reproduces the heuristic exactly, so a raw order vector
  can only match-or-lose), `random_sequential` train allocation, sigma schedules, and instances
  other than the three losing ones. Full per-run rows are in the TSV ledger
  `outputs/autoresearch/one_warehouse_multi_retailer_autoresearch/results.tsv` (33 rows).

## Code layout

- `env.rs` — raw state + `step_state` transition/cost (raw state quantities only; no
  normalization in the environment layer)
- `allocation.rs` — `proportional`, `random_sequential`, `min_shortage` rules
- `demand.rs` — Poisson / rounded-normal / discrete-uniform / deterministic demand
- `heuristics/echelon_base_stock.rs` — echelon base-stock order computation
- `finite_horizon_dp.rs` — reduced exact DP + heuristic / soft-tree evaluators
- `rollout.rs` — soft-tree rollout, feature construction, action modes
  (`direct_orders` / `echelon_targets` / `echelon_targets_with_alloc_targets` /
  `symmetric_echelon_targets` / `echelon_targets_with_holdback`). The holdback mode adds one
  SIGNED-residual control `h` (identity-leaf tail, `h == 0` at the zero-param warm-start =>
  byte-exact `echelon_targets` release) and rations against
  `release_capacity = max(warehouse_available − round(h).max(0), 0)`; the held-back units stay at
  the warehouse and feed the prob-0.8 partial-backorder emergency channel (central risk pooling)
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
