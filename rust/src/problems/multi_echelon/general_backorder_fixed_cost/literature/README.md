# Literature

This folder documents the public literature rows carried for `general_backorder_fixed_cost`
(the CardBoard Company general-network family).

## Canonical Reference

- **Geevers, van Hezewijk & Mes (2024)** — "Multi-echelon inventory optimization using deep
  reinforcement learning", *Central European Journal of Operations Research* **32(3):653-683**
  (published online first 19 July 2023; print Sept 2024). DOI:
  <https://doi.org/10.1007/s10100-023-00872-2>.
  Citation verified against Crossref (`api.crossref.org/works/10.1007/s10100-023-00872-2`),
  RePEc (`ideas.repec.org/a/spr/cejnor/v32y2024i3d10.1007_s10100-023-00872-2.html`) and the
  Springer landing page on 2026-05-31: title, authors (Kevin Geevers; Lotte van Hezewijk; Martijn
  R. K. Mes -- Crossref/SSRN record the Dutch tussenvoegsel "van", which RePEc's display drops),
  volume/issue/pages and DOI all match.
- Open MA thesis (the detailed open simulation description; **set-1 source**): Geevers (2020),
  "Deep Reinforcement Learning in Inventory Management", MSc thesis, University of Twente (work at
  ORTEC), <https://essay.utwente.nl/85432/> (PDF:
  <https://essay.utwente.nl/fileshare/file/85432/Geevers_MA_BMS.pdf>), Chapter 6 "The CardBoard
  Company". Title/author/institution verified 2026-05-31.

Related references cited in this folder (all verified to be real publications on 2026-05-31):

- **Kunnumkal & Topaloglu (2011)** — "Linear programming based decomposition methods for inventory
  distribution systems", *European Journal of Operational Research* **211(2):282-297**
  (`sciencedirect.com/.../S0377221710008064`). Source the thesis attributes the holding/backorder
  cost values to (0.6 / 1.0 / 19, no upstream backorder).
- **Chaharsooghi, Heydari & Zegordi (2008)** — "A reinforcement learning model for supply chain
  ordering management: an application to the beer game", *Decision Support Systems* **45(4):949-959**
  (`sciencedirect.com/.../S0167923608000560`). Source of the per-period event ordering.
- **De Kok, Grob, Laumanns, Minner, Rambau & Schade (2018)** — "A typology and literature review on
  stochastic multi-echelon inventory models", *European Journal of Operational Research*
  **269(3):955-983**, DOI 10.1016/j.ejor.2018.02.047. Typology used to classify the network.

The two CardBoard Company documents differ in scope for the general network:

- the **thesis** reports only the **set-1** benchmark (one order per stock point) and is the only
  OPEN source for any general-case number;
- the **journal** expands the general case into **three experiment sets** that differ in the
  agent/benchmark action space, and is the only source for the set 2 / set 3 rows. The journal full
  text is gated and the SSRN preprint (`ssrn.com/abstract=4227665`) returns 403, so the set 2 / set 3
  numbers below could NOT be confirmed against an open copy.

## Model (as implemented)

A two-echelon general network: 4 suppliers (unlimited) -> 4 warehouses (paper mills) -> 5 retailers
(corrugated plants); customer demand only at the 5 retailers; backorders (not lost sales); unit lead
times on every supplier->warehouse and warehouse->retailer edge.

- demand: Poisson, mean 15 per retailer per period
  - thesis Chapter 6 verbatim: "we decided to use a Poisson distribution with lambda = 15 to
    generate the demand" (this overrides the uniform[0,15] used in the earlier Beer-Game chapter).
- costs (Kunnumkal & Topaloglu 2011): warehouse holding 0.6, retailer holding 1.0, retailer
  customer-backorder 19.0, no warehouse backorder cost.
- routing (set 1): relative rationing — each retailer order is routed to exactly ONE upstream
  warehouse drawn according to the historical connection weights (Figure 6.1).
- warehouse allocation when short: serve the retailer with the lowest inventory position first
  (thesis: "The plant with the lowest inventory position will be fulfilled first").
- event order within a period: receive shipments -> suppliers ship warehouse orders -> warehouses
  fulfil current retailer orders -> retailers serve customer demand (excess -> customer backorders)
  -> clear existing backorders -> place next orders (matches `env.rs::advance_to_decision_state`).
- simulation window: the thesis (Sec. 6.6) measures the set-1 benchmark over a **50-period planning
  horizon, replicated 500 times**, and the thesis warm-up convention (Sec. 5.6) is a **25-period
  warm-up** on a 75-period run (first 25 removed). The repo's `references.rs` instead encodes
  `benchmark_periods=100, benchmark_warm_up_periods=50` (a 50-warm-up + 50-eval window). This is the
  **repo's own protocol choice**, NOT a value quoted from the open thesis; the set-1 reproduction
  (-1.1%) holds under it, but the field values should not be read as taken verbatim from the paper.
  (An earlier version of this README claimed the paper uses a 50-period warm-up; that was incorrect.)

## Published Benchmark Rows

The published general-network rows (one constant base-stock "benchmark" per set, plus the PPO
best/average over 10 runs):

| set | action space (paper)               | base-stock levels                        | benchmark cost | PPO best | PPO average |
|-----|------------------------------------|------------------------------------------|---------------:|---------:|------------:|
| 1   | order per stock point              | `[82,100,64,83, 35,35,35,35,35]`         | 10467          | 8714     | 630401      |
| 2   | order per edge                     | `[37,47,33,63, 30,30,30,30,30]`          | 4797           | 4175     | 314923      |
| 3   | order per edge, restricted transition | `[37,47,33,63, 30,30,30,30,30]`       | 4797           | 3935     | 4481        |

(First 4 levels = warehouses, last 5 = retailers.) The benchmark levels are tuned to a 98% fill-rate
target on the corrugated-plant / retailer connections; the paper reports general-case PPO improving
on the benchmark (the open journal abstract states ~6.6%; the SSRN preprint summary states ~17.5%
for the best runs — the references reflect the per-set best/average rows). The PPO "average" column
is dominated by the failed runs (5 of 10 runs do not learn correct order quantities), which is why
set-3's restricted transition (stable training) brings the average 4481 close to the best 3935,
while set-2's average 314923 does not.

## Current Verification Status

**Not literature-verified.** This problem is at best *partial*. Stated precisely:

- **Citations: literature-verified.** Every cited paper is real and its metadata
  (authors / year / title / venue / volume / issue / pages / DOI) is correct -- checked against
  Crossref, RePEc, Springer, ScienceDirect and the open University-of-Twente repository on
  2026-05-31 (see "Canonical Reference" above). No fabricated or mis-attributed citation was found.
- **Set-1 model + benchmark anchor: literature-verified (faithful) and reproduced within tolerance.**
  The set-1 inputs are confirmed verbatim against the open thesis Sec. 6.6: base-stock
  `[82,100,64,83,35,35,35,35,35]`, benchmark cost **10467**, 50-period horizon x 500 replications,
  Poisson(15) demand, costs (0.6 / 1.0 / 19, no upstream backorder) from Kunnumkal & Topaloglu
  (2011). The repo's node-base-stock simulation reproduces this at **~10355 (gap -1.1%)** with
  retailer fill 98-99% (the paper's target), so the published number is re-derived by a solver, not
  merely stored. CAVEAT: the repo evaluates over periods 50..100 (50-warm-up + 50-eval), whereas the
  open thesis uses a 50-period horizon with a 25-period warm-up; the gap is robust to this, but the
  window fields are the repo's protocol, not a value quoted from the paper.
- **Set-2 / set-3 rows: table-only AND not independently confirmed.** The numbers
  (benchmark **4797**, levels `[37,47,33,63,30,30,30,30,30]`, PPO 4175 / 3935, averages
  314923 / 4481) appear ONLY in the gated journal full text -- the open thesis covers set 1 only and
  SSRN returns 403, so they could not be confirmed against any open copy. They are stored as
  published anchors but are **NOT reproduced** by the repo: the configured per-edge routing gives
  ~15306 (gap +219%) at the published level 30. The exact per-edge inventory-position / order-up-to
  transition (and the set-3 "restricted transition") needed to reproduce 4797 is not in any open
  source and is not implemented.

Per-row reproduction status:

| instance                  | published | repo (configured mode)     | gap%    | status                         |
|---------------------------|----------:|----------------------------|--------:|--------------------------------|
| geevers2023_general_set1  | 10467     | ~10355 (single-connection) | -1.1%   | reproduced within tolerance    |
| geevers2023_general_set2  | 4797      | ~15306 (split by weight)   | +219%   | NOT reproduced (table-only)     |
| geevers2023_general_set3  | 4797      | ~15306 (split by weight)   | +219%   | NOT reproduced (set-2 mechanic)|

Numbers from `scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py`
(500 replications x 3 seeds for set 1; 500 replications for the sweeps). Set-1 fill rates land in
the 98-99% band at the retailers and ~98% at the warehouses, matching the paper's fill-rate target.

## Root Cause of the Set 2 / Set 3 Gap (precise)

Sets 2/3 use an **order-per-edge** action space: a separate order is placed on each
`(warehouse, retailer)` connection rather than one order per stock point that is then routed. The
published benchmark base-stock vector is still a **9-value per-stock-point** target. The open papers
state only that orders are placed "per edge"; the exact per-edge **inventory-position / order-up-to
transition** (and the set-3 "restricted transition function") is given only in the gated journal
full text, which could not be recovered from open sources.

A routing-mode + level sweep localises the gap to a **consistent ~6-7 unit offset in the retailer
order-up-to level** (set 2, evenly-split per-edge ordering, 500 reps):

| retailer order-up-to | repo cost | customer fill | warehouse fill (min) |
|---------------------:|----------:|--------------:|---------------------:|
| 30 (published)       | 11946     | 0.925         | 0.995                |
| 36                   | 4568      | 0.979         | 0.995                |
| 37                   | 4207      | 0.984         | 0.995                |
| 38                   | 3999      | 0.989         | 0.995                |
| 40                   | 3918      | 0.994         | 0.995                |

Retailer level **~36-37** reproduces BOTH the published cost (~4797) AND the paper's ~98% retailer
fill simultaneously, whereas the **published level 30** gives only ~92.5% fill / cost ~12000 in the
repo's convention. The offset is exactly what a different per-edge inventory-position convention
produces: the repo's `retailer_total_inventory_positions` (env.rs) nets the in-transit pipeline and
customer backorders into the order-up-to gap (standard), so a nominal target of 30 behaves as if it
were ~36-37 if the paper's order-per-edge transition does NOT net the per-edge pipeline the same way.
With Poisson(15) demand and lead time 1, an order-up-to of 30 cannot reach 98% fill in a standard
S-policy regardless of routing mode — confirming the gap is structural (the per-edge transition),
not a tuning artefact.

No implemented routing mode reproduces 4797 at the published level 30:

| routing mode                          | cost   | customer fill | warehouse fill (min) |
|---------------------------------------|-------:|--------------:|---------------------:|
| random_single_connection_by_weight    | 15601  | 0.901         | 0.677                |
| split_across_all_connections_by_weight| 15396  | 0.895         | 0.831                |
| split_across_all_connections_evenly   | 11946  | 0.925         | 0.995                |
| duplicate_target_all_connections      | 8746   | 0.991         | 0.763                |
| weighted_target_all_connections       | 22129  | 0.849         | 0.754                |

`duplicate_target` is the only mode reaching ~99% customer fill, but it over-stocks the retailers
(warehouse fill drops to ~76%) and costs 8746 — so even it does not match the paper's joint
(cost 4797, ~98% fill, low holding) signature. None of the modes matches both the cost and the
fill-rate target with the published level.

## What Was Changed

Earlier pass:

- `references.rs`: corrected the citation (was "Geevers et al. (2023) ... 32:157-187"; the
  authoritative RePEc/CEJOR record is **32(3):653-683, 2024**) and added the author list
  (Geevers, van Hezewijk & Mes). Rewrote the per-set `notes` to record the order-per-edge vs
  order-per-stock-point distinction, the reproduced/not-reproduced status, and the
  ~6-7 unit retailer-level verification target. The published numbers (10467, 4797) were NOT
  altered.
- created this `literature/README.md` and a verification-status note in the package README.
- added `scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py`
  (read-only benchmark + diagnostic harness).

Literature audit pass (2026-05-31, citation/verifiability only -- no logic, no numeric fields):

- Independently re-verified every citation against Crossref, RePEc, Springer, ScienceDirect and the
  open Twente repository; all metadata is correct. Confirmed the author tussenvoegsel "van Hezewijk"
  (Crossref/SSRN) is correct despite RePEc dropping it.
- Verified set-1 numbers verbatim against the open thesis (Sec. 6.6): levels, cost 10467,
  50-period x 500-rep protocol, Poisson(15), and the K&T-2011 cost values.
- **Corrected a factual prose error**: earlier text said the paper uses a "50-period warm-up +
  50-period evaluation window". The open thesis uses a 50-period horizon with a **25-period
  warm-up**; the repo's 50+50 window is the repo's own protocol choice, now labelled as such. The
  numeric fields (`benchmark_periods=100`, `benchmark_warm_up_periods=50`) were left unchanged
  (numeric edits are out of scope for this pass) -- see the package README / blockers.
- Made the verifiability statement explicit (citations literature-verified; set 1 faithful +
  reproduced within tolerance; set 2/3 table-only and not independently confirmable from open
  sources).

The benchmark protocol, demand, costs, set-1 routing, and the published levels were already
faithful for set 1 — they were not changed.

## Remaining Steps (deferred — risky / blocked)

1. Recover the journal's exact order-per-edge transition (and set-3 restricted transition) from the
   full text (gated). Then implement it as a new routing/transition mode in `heuristics.rs` so the
   published level 30 reproduces 4797 at ~98% fill. This is the single change that would flip sets
   2/3 from "carried" to "reproduced". It is deferred because the correct equation is not yet known
   and changing the existing `retailer_total_inventory_positions` convention would regress set 1.
2. Learned soft-tree benchmark: the `..._soft_tree_rollout` binding runs (depth-2 oblique/linear
   `vector_quantity` needs a 585-length flat-param vector for this network), but a trained vector is
   not checked in. Producing one requires a CMA-ES run through the `invman/` Python harness.

## Related Literature

- **Kunnumkal & Topaloglu (2011)** — source of the holding/backorder cost structure reused here.
- **Chaharsooghi, Heydari & Zegordi (2008)** — the Beer-Game linear case the thesis validated its
  simulator against (source of the per-period event ordering).
- **De Kok et al. (2018)** — typology used to classify the CardBoard Company network as "general".
