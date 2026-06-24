# General Backorder Fixed Cost

This subproblem carries the general-network CardBoard Company benchmark from Geevers, van Hezewijk &
Mes (2024) (online first July 2023) and the open MSc thesis Geevers (2020).

Important scope note:

- the folder name stayed from the earlier split decision
- the carried Geevers benchmark does **not** include fixed ordering costs in the published objective
- the implemented objective is holding cost plus backorder cost, exactly as written in Section 3 of the
  CEJOR paper and Chapter 6 of the thesis

## Formulation

We model a two-echelon general network with:

- 4 suppliers with unlimited capacity
- 4 warehouses
- 5 retailers
- customer demand only at the 5 retailers
- backorders, not lost sales
- constant unit lead time on supplier-to-warehouse and warehouse-to-retailer shipments
- Poisson retailer demand with mean `15`

The CardBoard Company network uses the historical warehouse-to-retailer connection weights shown in
Figure 6.1 of the thesis:

- retailer 0: `(0.60, 0.30, 0.10)` from warehouses `(0, 1, 3)`
- retailer 1: `(0.50, 0.40, 0.10)` from warehouses `(0, 1, 3)`
- retailer 2: `(0.15, 0.80, 0.05)` from warehouses `(0, 1, 3)`
- retailer 3: `(0.10, 0.80, 0.10)` from warehouses `(1, 2, 3)`
- retailer 4: `(0.70, 0.30)` from warehouses `(2, 3)`

Per-period costs:

- warehouse holding cost: `0.6`
- retailer holding cost: `1.0`
- warehouse-to-retailer backorder cost at warehouses: `0.0`
- retailer-to-customer backorder cost: `19.0`

The Rust environment follows the benchmark simulator structure from the thesis appendix:

1. Receive incoming deliveries.
2. Suppliers ship current warehouse orders.
3. Warehouses fulfill current retailer orders, ranking retailers by retailer inventory minus retailer
   customer backorders when stock is short.
4. Retailers satisfy current customer demand and accumulate customer backorders.
5. Remaining inventory is used to clear existing warehouse-to-retailer and retailer-to-customer backorders.
6. New upstream orders are placed for the next period.

The environment exposes raw state only. Any feature normalization belongs in rollout or policy code.

## Literature Rows

Published benchmark source:

- Geevers, van Hezewijk & Mes (2024), CEJOR `32(3):653-683` (online first 19 July 2023, print
  Sept 2024); DOI `10.1007/s10100-023-00872-2`. Citation metadata independently verified against
  Crossref, RePEc and Springer on 2026-05-31 (authors Kevin Geevers; Lotte van Hezewijk; Martijn
  R. K. Mes -- the Dutch "van" is correct per Crossref/SSRN).
- open MSc thesis (more detailed simulator; **only open source for any general-case number**):
  Geevers (2020), University of Twente, `essay.utwente.nl/85432` — reports only the set-1 benchmark.

The published-improvement figure differs between versions and the repo keeps both honestly: the
gated journal abstract reports general-case PPO ~6.6% over benchmark, while the SSRN preprint
summary reports ~17.5% for the best runs.

The general case has three experiment sets that differ in the **action space**:

- set 1 — one **order per stock point** (relative-rationing routing to a single upstream edge);
- set 2 — one **order per edge**;
- set 3 — one **order per edge with a restricted transition function** (training-side change only;
  same benchmark base-stock policy as set 2).

Published rows (the constant base-stock "benchmark", plus PPO best/average over 10 runs):

- set 1: benchmark `10,467`, levels `[82, 100, 64, 83, 35, 35, 35, 35, 35]`, PPO best `8,714`,
  PPO average `630,401`
- set 2: benchmark `4,797`, levels `[37, 47, 33, 63, 30, 30, 30, 30, 30]`, PPO best `4,175`,
  PPO average `314,923`
- set 3: benchmark `4,797`, levels `[37, 47, 33, 63, 30, 30, 30, 30, 30]`, PPO best `3,935`,
  PPO average `4,481`

(First 4 levels = warehouses, last 5 = retailers.) The benchmark levels are tuned to a 98%
fill-rate target on the corrugated-plant / retailer connections. Note: the thesis Figure 6.6 shows
the benchmark deliberately holds the **warehouse (paper-mill) fill rate lower** (about 53-74% at the
paper mills) while keeping the **retailer (corrugated-plant) fill at ~97-99%** — only the
customer-facing stock points are held to 98%. (A previous version of this README claimed ~98-99% at
all warehouses; that was incorrect.)

See `literature/README.md` for the full literature audit and the verification target.

## Current Verification Status

This subproblem is **partly literature-verified by executing checks**. The set-1 CardBoard row and
the Kunnumkal-Topaloglu divergent row reproduce published constant node-base-stock benchmarks within
tolerance; set 2 and set 3 remain table-only debt because the gated paper's order-per-edge
transition is not implemented.

Benchmark results (node-base-stock heuristic at published levels, configured routing mode), from
`scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py`
(500 replications x 3 seeds for set 1, 500 reps for the sweeps):

| instance | published | repo  | gap%   | status                              |
|----------|----------:|------:|-------:|-------------------------------------|
| set 1    | 10467     | 10355 | -1.1%  | reproduced within tolerance         |
| set 2    | 4797      | 15306 | +219%  | NOT reproduced (table-only anchor)  |
| set 3    | 4797      | 15306 | +219%  | NOT reproduced (set-2 mechanic)     |
| KT divergent | 4059  | 3931  | -3.0%  | reproduced within tolerance         |

(The set-1 published anchor is confirmed verbatim in the open thesis; the set-2/set-3 anchors exist
only in the gated journal full text and could not be confirmed against an open copy.)

Set 1 also reproduces the paper's customer-facing fill-rate target: retailer fill is in the
98-99% band. The warehouse / paper-mill fill rate is intentionally lower in the thesis figure, so it
is not a 98% verification target.

### Root cause of the set 2 / set 3 gap (precise)

Sets 2/3 use an **order-per-edge** action space (one order per `(warehouse, retailer)` connection),
but the published benchmark base-stock is still a 9-value per-stock-point target. The open papers
state only that orders are placed "per edge"; the exact per-edge inventory-position / order-up-to
transition (and the set-3 "restricted transition function") is given only in the **gated journal full
text**, which could not be recovered from open sources.

A routing-mode + level sweep localises the gap to a **consistent ~6-7 unit offset in the retailer
order-up-to level**. Under evenly-split per-edge ordering (set 2, 500 reps):

| retailer order-up-to | repo cost | customer fill | warehouse fill (min) |
|---------------------:|----------:|--------------:|---------------------:|
| 30 (published)       | 11946     | 0.925         | 0.995                |
| 36                   | 4568      | 0.979         | 0.995                |
| 37                   | 4207      | 0.984         | 0.995                |
| 40                   | 3918      | 0.994         | 0.995                |

Retailer level **~36-37** reproduces BOTH the published cost (~4797) AND the paper's ~98% retailer
fill, while the **published level 30** gives only ~92.5% fill / cost ~12000 in the repo's convention.
With Poisson(15) demand and lead time 1, a per-edge order-up-to of 30 cannot reach 98% fill in a
standard S-policy under any implemented routing mode — so the gap is structural (the per-edge
transition convention), not a tuning artefact. The repo's `retailer_total_inventory_positions`
(env.rs) nets the in-transit pipeline and customer backorders into the order-up-to gap (standard); a
nominal target of 30 behaves as ~36-37 if the journal's order-per-edge transition does not net the
per-edge pipeline the same way.

No implemented routing mode reproduces 4797 at the published level 30 (see `literature/README.md`).

### Honest status

- **citations: literature-verified** — every cited paper is real with correct metadata
  (checked against Crossref / RePEc / Springer / ScienceDirect / Twente, 2026-05-31);
- **set 1: faithful + reproduced within tolerance (-1.1%)** — inputs confirmed verbatim against the
  open thesis Sec. 6.6 (levels `[82,100,64,83,35,35,35,35,35]`, cost 10467, 50-period x 500-rep,
  Poisson(15), K&T-2011 costs), and the published 10467 is re-derived by the repo solver (~10355);
- **Kunnumkal-Topaloglu divergent: reproduced within tolerance (-3.0%)** — the open thesis reports
  the 1-warehouse / 3-retailer base case at levels `[124,30,30,30]`, cost 4059, and the repo's
  resampled-uniform-Poisson demand path re-runs it at ~3931;
- **set 2 / set 3: table-only and NOT independently confirmable** — the numbers (4797, 4175, 3935,
  averages 314923 / 4481, levels `[37,47,33,63,30,...]`) appear only in the gated journal full text
  (the open thesis has set 1 only; SSRN returns 403) and are carried as published rows. The per-edge
  transition that would reproduce 4797 at the published level 30 is not implemented (its exact spec
  is in the gated journal); implementing it is the single change that would flip sets 2/3 to
  "reproduced". Deferred because the correct equation is not yet known and altering the existing
  inventory-position convention would regress set 1.

WINDOW CAVEAT: the repo evaluates the benchmark over periods 50..100 (`benchmark_periods=100`,
`benchmark_warm_up_periods=50`, a 50-warm-up + 50-eval window). The open thesis instead uses a
50-period planning horizon with a 25-period warm-up (Sec. 5.6 / 6.6). The 50+50 window is the repo's
own protocol choice, not a value quoted from the paper; the set-1 reproduction is robust to it.

## General-network DIVERGENT instance (Kunnumkal & Topaloglu) — VERIFIED

A second verified row for this family, added 2026-06-05. Reference name
`kunnumkal_topaloglu_divergent`. This is the Kunnumkal & Topaloglu (2011) divergent base case as
reported in the open Geevers (2020) thesis (essay.utwente.nl/85432, Ch. 5).

DISAMBIGUATION: this is **not** the 4-warehouse / 5-retailer CardBoard network of set 1/2/3, and it
is **distinct** from the paper's Section-6 Gijsbrechts/Van-Roy "divergent special-delivery" model.
Call it the **general-network divergent (Kunnumkal–Topaloglu)** instance.

- topology: 1 supplier → 1 warehouse → 3 identical retailers (3 unit-weight edges, so routing is
  degenerate — the whole order goes down the single edge);
- costs: the SAME K&T-2011 costs as set 1 (warehouse holding `0.6`, retailer holding `1.0`, retailer
  backorder `19.0`, no warehouse backorder);
- lead time: constant `1` everywhere;
- demand: Poisson(α) with **α ~ Uniform[5,15] resampled every period per retailer** (nonstationary
  mean); warehouse demand = sum of retailer demands;
- protocol: horizon 50, 25-period warm-up applied to a 75-period run (`benchmark_periods=75`,
  `benchmark_warm_up_periods=25`), 1000 replications;
- published constant node-base-stock benchmark `4,059` at levels `[warehouse 124, retailers 30/30/30]`
  (confirmed open); published DRL `3,724` (cross-protocol context, not a head-to-head beat).

### Env change supporting this instance (the `demand_mode` flag)

The simulator/rollout previously used a single fixed Poisson mean. A `DemandMode` selector now lives
on the reference instance (`demand_mode`, `demand_alpha_min`, `demand_alpha_max`):

- `fixed_poisson` — every retailer draws `Poisson(retailer_demand_mean)` every period. This is the
  byte-for-byte original path; **sets 1/2/3 keep it** and set 1 still reproduces ~10,355.
- `resampled_uniform_poisson` — every period and retailer first draws a fresh
  `α ~ Uniform[demand_alpha_min, demand_alpha_max]`, then draws `Poisson(α)` (nonstationary mean).
  Only the divergent instance uses it.

The single branch point is `heuristics::sample_period_demands`, used by both the benchmark simulator
(`simulate_node_base_stock_policy*`) and the policy `rollout`.

### Verification (executing)

`tests::verification::divergent_kt_benchmark_reproduces_published_cost` re-runs the env's constant
node-base-stock gate at `[124,30,30,30]` under `resampled_uniform_poisson` and the thesis protocol
(1000 reps) and asserts the mean is within 5% of the published `4,059`.

| instance                       | published | repo (mean) | gap%   | retailer fill | status                      |
|--------------------------------|----------:|------------:|-------:|--------------:|-----------------------------|
| kunnumkal_topaloglu_divergent  | 4059      | ~3,931      | -3.0%  | ~98.6%        | reproduced within tolerance |

(3 seeds × 1000 reps; same direction/magnitude as set 1's -1.1% simulator residual.) This is why
`literature_verified = true` for this instance.

### Learned-policy result (full budget, 2000 held-out CRN seeds)

`scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py
--reference kunnumkal_topaloglu_divergent --budget full` (node-base-stock-targets soft tree,
warm-started at `[124,30,30,30]`, depth-2 oblique constant leaf):

| seed | warm-start gen0 | learned held-out | vs repo heuristic (3,930) | vs published 4,059 | vs published DRL 3,724 | verdict |
|-----:|----------------:|-----------------:|--------------------------:|-------------------:|-----------------------:|---------|
| 123  | 3,936.5         | **2,469.1 ± 7.6**| -1,461 (-37.2%)           | -1,590             | -1,255                 | beats   |
| 777  | 3,936.5         | **2,477.9 ± 8.0**| -1,453 (-37.0%)           | -1,581             | -1,246                 | beats   |
| 9101 | 3,936.5         | **2,488.4 ± 7.9**| -1,442 (-36.7%)           | -1,571             | -1,236                 | beats   |
| 9102 | 3,936.5         | **2,498.4 ± 7.6**| -1,432 (-36.4%)           | -1,561             | -1,226                 | beats   |
| 9103 | 3,936.5         | **2,489.0 ± 6.8**| -1,441 (-36.7%)           | -1,570             | -1,235                 | beats   |

All five seeds beat the reproduced constant node-base-stock benchmark by about 36.8% (well beyond
SEM, genuine out-of-sample) and land below the published DRL 3,724. The DRL number is cross-protocol
context, not a like-for-like beat; the gate is the repo's own reproduced constant base-stock
benchmark.
