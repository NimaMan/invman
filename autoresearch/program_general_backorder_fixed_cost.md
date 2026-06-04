# general-backorder-fixed-cost autoresearch (Geevers set-1 general network)

This is the general-network multi-echelon counterpart to the dual-sourcing,
multi_echelon, one_warehouse_multi_retailer and joint_replenishment autoresearch
programs. It targets the `multi_echelon/general_backorder_fixed_cost` problem
(Geevers, van Hezewijk & Mes 2024, *Central European Journal of Operations Research*
32(3):653-683, online first 2023; CardBoard Company general network), experiment **set 1**
(`geevers2023_general_set1`): 4 suppliers / 4 warehouses / 5 retailers, Poisson(15) retailer
demand, unit lead times, backorders (no fixed ordering cost in the published objective --
holding + backorder only). The env is the verified family member
(`rust/src/problems/multi_echelon/general_backorder_fixed_cost/`, set-1 reproduction asserted
in-crate by `tests::verification::set1_benchmark_reproduces_geevers_published_cost`).

The single-policy loop is the same shape as the sibling programs: train ONE soft-tree
CMA-ES policy on the NAMED set-1 instance, warm-started so generation 0 reproduces the
published constant node-base-stock benchmark, evaluate held-out CRN cost + gap vs the
benchmark, append a TSV ledger row. The runner is
`scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py`; it trains
PURELY in Python against the installed `invman_rust` via
`multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout` (no rebuild).

## Benchmark

INSTANCE = `geevers2023_general_set1` (carried verbatim in `references.rs`). Routing mode
`random_single_connection_by_weight` (set 1 = one order per stock point, relative-rationing
routing to a single upstream connection). 100 periods, 50-period warm-up, 500 reps.

BASELINES (from `references.rs`):
- **published constant node-base-stock benchmark = 10,467** at levels
  `[82,100,64,83,35,35,35,35,35]` (4 warehouses + 5 retailers) -- the row the learned policy
  must beat. The repo simulator reproduces **~10,355** (gap -1.1%, 3 seeds x 500 reps); this
  reproduction is the keep/discard GATE.
- **published PPO best = 8,714** -- the DRL target to approach/beat (reported alongside,
  not the gate). The published PPO *average* (630,401) is degenerate and ignored.

Evaluation protocol: held-out common-random-number base seeds from `500_000` (stride 1,000),
disjoint from the training block (`10_000`, stride 1,000). Each rollout is one 100-period
path with the 50-period warm-up cut; full budget averages 2,000 held-out paths. The SAME
seed block scores the warm-start (= benchmark policy) and the learned policy (paired).

## The action geometry (the policy)

The rollout binding's `node_base_stock_targets` action mode interprets the soft tree's 9-dim
`vector_quantity` output as the per-node **order-up-to (base-stock) target levels** (4
warehouses + 5 retailers); the env's order-up-to + relative-rationing routing converts those
targets into orders. This is the heuristic's own coordinate system:

- A **state-independent** soft tree (split weights = 0, all leaves equal) emits a CONSTANT
  target vector == exactly a constant node-base-stock policy. Encoding the published levels in
  the leaf parameters makes **generation 0 reproduce the published benchmark** -- the same
  gen-0-reproduces-heuristic device used for `symmetric_echelon_targets` (OWMR) and
  `capped_dual_index` (dual_sourcing).
- A **state-dependent** soft tree lets the per-node order-up-to levels react to the compact
  inventory-position summary (warehouse + retailer inventory positions, totals, demand mean,
  remaining-horizon fraction) -- a strictly richer class than any single constant base-stock
  vector. This is where the learned policy wins.

WARM-START encoding (gen-0 == benchmark):
- constant leaf: `scaled = min + sigmoid(p)*(max-min)` => `p_i = logit((L_i-min_i)/(max_i-min_i))`.
- linear leaf: `scaled = min + softplus(bias + w.state)`; `w=0`, `bias_i = ln(exp(L_i-min_i)-1)`.
- split weights/bias = 0 (50/50 gating, all leaves identical => constant).

Action box (physical caps, interior to the operating region): warehouses [0,220], retailers
[0,140] -- well above the published max (100) so the learned policy's higher targets are
reachable.

## Intended search surface (editable levers)

- Soft-tree structure: `--depth` (1,2), `--leaf_type` (constant/linear), `--temperature`,
  split type (oblique).
- CMA warm-start `sigma_init`: **small** (0.2-0.3). A large sigma (>=3) saturates the
  sigmoid leaves and lets the oblique split weights run wild -> divergence to ~30k; sigma 0.2
  refines around the benchmark to ~8.0k. This is the documented "warm-start saturation"
  failure mode -- re-anchor and shrink sigma, do not add optimizer budget.

## Budgets (in the runner)

- `smoke`: popsize 8 x 8 gen, 4 train / 64 eval seeds (wiring check).
- `screening`: popsize 16 x 40 gen, 8 train / 256 eval seeds.
- `full`: popsize 24 x 80 gen, 12 train / 2,000 eval seeds, sigma 0.20.

HARD CPU CAP: the runner sets `RAYON_NUM_THREADS`/`OMP_NUM_THREADS` to 2 before importing
`invman_rust` (the population rollout fans out via rayon, not a Python process pool).

## Autoresearch outcome (what we know)

Commit `01c657a`, `geevers2023_general_set1`, full budget (popsize 24 x 80 gen, 12 train /
2,000 held-out CRN seeds, sigma 0.20), depth-2 oblique **constant-leaf**
`node_base_stock_targets`, warm-started at the published levels `[82,100,64,83,35x5]`:

| policy | held-out mean cost | vs repo heuristic (10,355) | vs published 10,467 | vs PPO best 8,714 |
|---|---|---|---|---|
| warm-start (gen 0 = benchmark) | 10,378.6 +/- 10.6 | +0.2% | -0.8% | +1,664 |
| **learned soft tree** | **8,034.8 +/- 17.6** | **-22.4%** | **-2,432** | **-679** |

The learned policy **beats the constant node-base-stock benchmark by 22.4%** (>> 2x SEM, a
genuine out-of-sample win, not eval-seed noise) and lands **679 below the published PPO best
(8,714)**, i.e. it surpasses the paper's DRL row on this set-1 instance. Generation 0
reproduces the benchmark (10,378.6 ~ repo 10,354.8), so the warm-start guarantee holds and the
2,320-cost improvement is what CMA-ES added on top.

A second full run with a different CMA seed (777) reached **7,590.7 +/- 19.2**
(-26.7% vs heuristic, -1,123 vs PPO best) -- both independent seeds beat the benchmark by
>22% and land below the PPO best, so the win is robust to CMA initialization, not seed-luck.

Screening probes (256 eval seeds) ranked the structures: depth-1 constant sigma-0.2 ~8,095,
depth-1 constant sigma-0.3 ~8,183, depth-2 constant sigma-0.3 ~8,061. depth-2 constant won and
was promoted to full. The linear-leaf head (585 params vs 81 for depth-2 constant) is far less
sample-efficient and only matched the benchmark under the smoke budget; the compact constant
leaf is the operative design. **Lesson (consistent with the project thesis):** putting the
policy in the benchmark's order-up-to coordinate system and letting a state-dependent tree
modulate the per-node targets -- not a bigger/raw action space -- is the lever.

Artifacts: `outputs/autoresearch/general_backorder_fixed_cost_autoresearch/`
(`results.tsv`, per-run JSON incl. the trained 81-dim parameter vector).
