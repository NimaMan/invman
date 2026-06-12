# `invman/benchmarks/runners/` — the executable baseline layer

This package is the **executable** half of the benchmark surface. The catalog
(`invman.benchmarks.catalog`) tells you *what* the reference problems are and
*what numbers* the literature reports, purely from
`docs/benchmarks/BENCHMARK_MANIFEST.json`. The runners let you **run** them: load
a literature instance, re-run its baseline on the live env, and score your own
policy on the same instance under the same protocol.

It implements `PROPER_REPO_BUILD_PLAN.md` workstream **(a)** ("standard benchmark
API per problem") on top of the Rust reference accessors that already exist in
`invman_rust` (`<problem>_list_reference_instances` /
`<problem>_get_reference_instance` / the `*_search_from_demands` and
`*_soft_tree_rollout` bindings).

## The one thing to know

```python
from invman.benchmarks import catalog

inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")

inst.params              # env parameters of the literature instance (free)
inst.published_costs     # {'optimal': 4.73, 'myopic2': 4.82, ...}   (free)
inst.reference_cost      # 4.73  — the single canonical number to beat
inst.run_baselines()     # RE-RUN the shipped baselines on the live env
inst.evaluate(my_params) # score YOUR soft-tree policy on this instance
inst.compare(my_cost)    # signed gap vs the reference (+ a 'beats' verdict)
```

`catalog.get(problem).load_instance(name)` and `runners.load_instance(problem,
name)` are equivalent. `name=None` returns the family's primary instance.

## Coverage — all 14 catalog families (157 reference instances)

`base.py` is the shared vocabulary: `EvalProtocol` (the comparison contract),
`Baseline` (one comparator cost + provenance: `is_published` / `is_optimal` /
`is_reference`), `ReferenceInstance` (the handle a consumer holds), `ProblemRunner`
(abstract per-family driver). Two class flags shape a runner: `supports_evaluate`
(is the soft-tree rollout in the CMA-ES seam?) and `lower_is_better` (cost vs
profit). `__init__.py` is the lazy registry.

Every family supports `list_instances` / `load_instance` / `published_baselines`
/ `reference_cost` / `run_baselines` / `compare`. Three additionally support
`evaluate` (soft-tree in the `build_policy` + `get_model_fitness` seam); the rest
set `supports_evaluate=False` and `evaluate()` raises an actionable error.

**Literature-verified default.** `runners.available_runners()` (and
`catalog.list_problems(literature_verified=True)`) return only the **9
literature-verified** families — those whose env reproduces a real literature
anchor (`verification_tier != 'faithful'`). The **5 `faithful`** families (the
`❌` rows below: `one_warehouse_multi_retailer`, `joint_pricing_inventory`,
`procurement_removal_inventory`, `random_yield_inventory`,
`vendor_managed_inventory`) are repo-native `<author>_style` / paywalled instances
solved by the repo's own DP — **not** a reproduction of any published number — so
they are hidden by default. They stay fully usable via `get_runner(name)` /
`load_instance(name)` and `available_runners(include_unverified=True)`; each
instance self-reports `inst.literature_verified` / `inst.verification_tier`. The
partition, with file:line evidence (adversarially verified, 28 agents), is in
[`../../../docs/benchmarks/LITERATURE_VERIFICATION_AUDIT_2026_06_12.md`](../../../docs/benchmarks/LITERATURE_VERIFICATION_AUDIT_2026_06_12.md).

| Runner file | Family | Inst. | `evaluate` | Lit-verified | Reference / `run_baselines` |
|---|---|---:|:---:|:---:|---|
| `lost_sales_runner` | lost_sales (vanilla + fixed_order_cost) | 34 | ✅ | ✅ strict | optimal/myopic/svbs/capped · exact (s,S)/(s,nQ)/modified |
| `dual_sourcing_runner` | dual_sourcing (Gijs Fig-9) | 6 | ✅ | ✅ reference | published gaps + 4 heuristic searches (capped-dual-index proxy) |
| `multi_echelon_runner` | multi_echelon (divergent Van Roy / Gijs) | 5 | ✅ | ✅ mixed | best constant base-stock (grid-widened to the published optimum) |
| `perishable_inventory_runner` | perishable_inventory (De Moor / Farrington) | 32 | — | ✅ strict | discounted-return VI optimum + base-stock search |
| `spare_parts_inventory_runner` | spare_parts_inventory (Kranenburg) | 35 | — | ✅ strict | Kranenburg analytical optimum (Table 5.2; different model from the env) |
| `joint_replenishment_runner` | joint_replenishment (Vanvuchelen) | 16 | — | ✅ reference | exact-DP optimum + published action anchor q=(0,6) |
| `nonstationary_lot_sizing_runner` | nonstationary_lot_sizing (Dehaybe) | 8 | — | ✅ reference | rolling-DP (s,S) + simple (s,S) re-sim (companion CSV) |
| `ameliorating_inventory_runner` | ameliorating_inventory (Pahr–Grunow) | 2 | — | ✅ reference | perfect-information LP **profit** bound (`lower_is_better=False`) |
| `decentralized_inventory_control_runner` | decentralized_inventory_control (Beer Game) | 1 | — | ✅ reference | Sterman closed-form 204 (+ honest env.rs 378/278 split) |
| `one_warehouse_multi_retailer_runner` | one_warehouse_multi_retailer (Kaynov) | 14 | — | ❌ faithful | exact-DP anchor + echelon base-stock (approx-only, no published number) |
| `joint_pricing_inventory_runner` | joint_pricing_inventory | 1 | — | ❌ faithful | exact-DP on a repo-native `zhou2022_style` instance |
| `procurement_removal_inventory_runner` | procurement_removal_inventory (Maggiar) | 1 | — | ❌ faithful | exact-DP on a `maggiar2017_style` instance (no published row) |
| `random_yield_inventory_runner` | random_yield_inventory (Yan) | 1 | — | ❌ faithful | exact-DP on a `yan2026_style` instance (no public benchmark) |
| `vendor_managed_inventory_runner` | vendor_managed_inventory (Sui–Gosavi–Lin) | 1 | — | ❌ faithful | base-stock heuristics (paper paywalled; only a handout) |

## How `evaluate` stays honest

`evaluate` does not re-implement a rollout. It reuses the **exact CMA-ES training
seam** — `invman.config.get_config` → set env fields from the instance →
`apply_policy_name` → `invman.policy_build.build_policy` →
`invman.rollout_fitness.get_model_fitness`. So a policy scored here is scored by
byte-identical code to the optimizer's fitness; there is no second, drifting
evaluator. Size your weight vector with `inst.policy_param_count(**structure)`.

The shipped policy class is the **soft tree** (the repo's workhorse); a different
method's policy is compared via the reported numbers (`run_baselines` +
`published_costs`), which is what the manifest cards surface.

## Provenance discipline

Every `Baseline` records its `source` and whether it is `is_published` (a paper
number) vs recomputed on the live env, and whether it is the exact optimum
(`is_optimal`) or the family's declared canonical comparator (`is_reference`).
Published numbers are never silently conflated with recomputed ones — the same
honesty rule as `docs/benchmarks/VERIFICATION_LEDGER.md`.

## Coverage today / next

**All 14 catalog families have a runner** (`runners.available_runners()` is the
live list). What is NOT yet covered:

- **`evaluate` for the metadata-only 11** — their soft-tree rollouts exist in
  `invman_rust` but are not in the `build_policy` + `get_model_fitness` seam, so
  policy scoring raises. Wiring a family is: add a `build_policy` + a
  `rollout_fitness` branch, then set `supports_evaluate=True` + implement
  `_eval_model_and_args`.
- **The other `multi_echelon` subfamilies** (serial / assembly / general-backorder-
  fixed-cost / PADN) — the divergent runner covers the Van Roy / Gijs settings;
  the rest have their own accessors and are a natural extension of that runner.
