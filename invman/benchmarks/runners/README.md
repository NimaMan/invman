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

## Files (functionality)

| File | What it does |
|---|---|
| `base.py` | Shared vocabulary: `EvalProtocol` (seeds/horizon/warm-up — the comparison contract), `Baseline` (one comparator cost + provenance), `ReferenceInstance` (the handle a consumer holds), `ProblemRunner` (abstract per-family driver; implements the family-INDEPENDENT load + multi-seed `evaluate` + param-count discovery). |
| `lost_sales_runner.py` | `lost_sales` family — **vanilla** (Zipkin grid, 33 instances; baselines optimal/myopic1/myopic2/svbs/capped) **+ fixed_order_cost** (Bijvank 2015; baselines optimal-DP/(s,S)/(s,nQ)/modified). One runner, both subfamilies. |
| `dual_sourcing_runner.py` | `dual_sourcing` — the 6 Gijsbrechts (2022) Figure-9 instances. Published **gaps** (capped-dual-index 0%, A3C 0.52%, …) are free; `run_baselines` grid-searches the 4 heuristics on a fixed demand path to get the **absolute** costs. |
| `multi_echelon_runner.py` | `multi_echelon` **divergent** subfamily — the 5 Van Roy / Gijs setting instances. `run_baselines` grid-searches the best **constant base-stock** (the canonical comparator), widening the grid to span the published optimum on the Van Roy reproduction rows. |
| `__init__.py` | Registry: `get_runner(problem)`, `load_instance(problem, name)`, `available_runners()`. Lazy — importing `catalog` never imports a runner or `invman_rust`. |

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

Runners exist for `lost_sales` (+fixed), `dual_sourcing`, and the `multi_echelon`
divergent subfamily — the families with a `*_search_from_demands` /
constant-base-stock baseline and a soft-tree rollout. The other `multi_echelon`
subfamilies (serial / assembly / general-backorder-fixed-cost / PADN) and the
remaining 10 catalog families have their reference accessors in `invman_rust`
already; wiring each into a `ProblemRunner` (same four hooks) is the next
increment. `runners.available_runners()` is the live list.
