<!--
ALGORITHMIC / CONCEPTUAL DESCRIPTION (read first)

PURPOSE. This document is the conceptual spine of the invman benchmark — the "ImageNet for
inventory control" taxonomy. It mirrors the 7 structural questions the repo's own FlowNet
language already asks of every problem (src/problems/core/flownet/question.rs), enumerates the
SPACE OF ANSWERS each question admits across inventory control, then maps which fundamental
answers the 14 implemented benchmark families cover and which are still missing.

HOW IT IS DERIVED (so it can be regenerated mechanically):
  1. The 7 questions are verbatim from FlowNetQuestion::prompt (question.rs).
  2. The "answer space" per question is the union of the enum variants the repo's core FlowNet
     layers can express — Topology / StockRole / FlowMode (physical), StochasticProcess
     (stochastic), ActionShape / ObservationMode / ShortageReaction / IssuanceRule (control),
     ObjectiveTerm / RewardConvention / PerformanceMetric (objective), Stage / TimingConstraint
     (timing) — plus the concrete attribute strings the family formulations instantiate.
  3. Each family's row in the coverage table is read off its src/problems/<fam>/BENCHMARK.md
     "One-line MDP" + its flownet/formulation.rs where one exists.
  4. Difficulty + verification tier are taken from BENCHMARK_MANIFEST.json and
     VERIFICATION_LEDGER/README.md (the ledger wins on any conflict; see "Honest verification" below).

This file does NOT define new code. It is a map. The authoritative machine-readable index is
BENCHMARK_MANIFEST.json; the authoritative honesty record is VERIFICATION_LEDGER/README.md; the
expansion order / decision rule is in docs/literature/fundamental_problem_families/README.md.
-->

# The Fundamental Questions of Inventory Control

**Goal — an ImageNet for inventory control.** ImageNet did not advance vision by collecting one
more clever classifier; it advanced it by laying down a *fixed, broad, honestly-labeled task
surface* that every method could be measured on. This benchmark aims to be that surface for
inventory control: a library of structurally distinct control problems, each faithful to a
literature model, each scored by the same Rust rollout oracle, each labeled with an honest
verification provenance.

To build such a surface you first need to know **what makes two inventory problems different**.
This repo already answers that question structurally. Every problem in `invman` is a
`FlowNetFormulation` (`src/problems/core/flownet/formulation.rs`) that must answer **seven
questions** (`src/problems/core/flownet/question.rs`). Those seven questions are the coordinate
axes of the problem space. A "fundamental inventory problem" is just a distinct, literature-grounded
point in that space — a distinct *combination of answers*.

This document (1) restates the 7 questions, (2) enumerates the **space of answers** each one admits,
(3) maps which answers the 14 implemented families cover, (4) assesses which fundamental axes are
well-covered vs thin, and (5) proposes candidate new fundamental problems that fill the gaps.

---

## 1. The 7 fundamental questions (verbatim)

From `FlowNetQuestion::prompt` — these characterize *any* inventory-control problem:

| # | FlowNet question | Repo core layer that answers it |
|---|---|---|
| Q1 | **What inventory states exist?** | `PhysicalLayer` — `Topology`, `StockNodeSpec`/`StockRole` |
| Q2 | **How can material move or transform?** | `PhysicalLayer` — `PipelineSpec`, `FlowEdgeSpec`/`FlowMode` |
| Q3 | **What random events occur?** | `StochasticLayer` — `StochasticProcess` |
| Q4 | **What can the controller choose?** | `ControlLayer` — `ActionSpec`/`ActionShape` |
| Q5 | **What can the controller observe, and when?** | `ControlLayer` — `ObservationSpec`/`ObservationMode` |
| Q6 | **How is performance scored?** | `ObjectiveLayer` — `ObjectiveTerm`, `RewardConvention`, `PerformanceMetric` |
| Q7 | **What timing rules and feasibility constraints shape the system?** | `TimingLayer` — `Stage` schedule, `TimingConstraint`; `ControlLayer::ServiceSpec`/`ShortageReaction`/`IssuanceRule`; `FeasibilityConstraint` |

A formulation is well-posed only when it answers **all seven** (`answers_all_questions`).

---

## 2. The axis space — the space of answers each question admits

The values below are the union of what the repo's core FlowNet enums can express plus the concrete
attributes the families instantiate. This is the "menu"; a fundamental problem is a selection from it.

### Q1 — What inventory states exist? (`Topology` × `StockRole`)
- **Topology**: `SingleLocation` · `SerialChain` · `DivergentNetwork` · `DirectedNetwork` (general acyclic) · `JointMultiItem`.
- **StockRole** (what each stock node *means*): `OnHand` · `Pipeline` (in-transit) · `Backlog` · `AgeBucket` (shelf-life / amelioration age profile) · `WorkInProcess` · `Reserve` · `SupplySource` · `DemandSink` · `Custom` (e.g. *returnable pool*, *waste sink*, *repair pool*).
- **State augmentations beyond stock**: realized **price** state (ameliorating, joint-pricing); **forecast window** (nonstationary); **regime/Markov-modulating** index (MMPP demand); installed-base / failure state (spare parts).

### Q2 — How can material move or transform? (`FlowMode`)
`Procurement` · `Shipment` (echelon-to-echelon) · `DemandFulfillment` · `Aging` (shelf-life or amelioration progression) · `Transformation` (production / assembly merge) · `Repair` · `Removal` (waste, vendor-return, liquidation) · `Return` (closed-loop) · `Custom`. Movement is also shaped by **lead time** (single inbound pipeline vs per-echelon pipelines) and by **emergency / expedited channels** (dual-source expedite, OWMR special delivery).

### Q3 — What random events occur? (`StochasticProcess`)
- **Demand**: i.i.d. `Poisson` / `Geometric` (high-CV) / rounded-Gamma / CV-Normal; **nonstationary forecast-driven**; **Markov-modulated** (`MarkovModulatedPoisson2`); **price-elastic** (demand a function of the chosen price).
- **Supply-side**: `Yield` (all-or-nothing / random yield); `Disruption` / `TransitDelay` (stochastic or endogenous lead time); `Failure` (spare-parts unit failures, a binomial demand-for-spares).
- **Return-side**: `ReturnArrival` (closed-loop returns/remanufacturing) — *enum exists, no family instantiates it yet*.
- **Price/cost**: stochastic **purchase price** (ameliorating).

### Q4 — What can the controller choose? (`ActionShape`)
`ScalarOrder` (one item, one quantity) · `VectorOrder` (multi-item / multi-node, e.g. joint replenishment, PADN pairwise order-up-to) · `DualSourceOrderPair` (regular + expedited split) · `Allocation` (divide scarce upstream stock across retailers) · `Routing` · `PurchaseAndRemoval` (order **and** dispose, procurement-removal) · order **+ price** (joint-pricing, a scalar order paired with a discrete price index) · `Custom`. Multi-agent variants have **one action per decentralized stage** (Beer Game).

### Q5 — What can the controller observe, and when? (`ObservationMode`)
`FullState` (centralized; sees all echelons/ages/pipelines) · `LocalState` (decentralized; each agent sees only its own stock/backlog/orders) · `ReducedState` (sufficient-statistic compression, e.g. dual-sourcing reduced post-decision vector) · `ForecastAugmented` (state includes a forward demand-forecast window) · `Delayed` · `Custom`. The *timing* of observation (start-of-period, after-receipts) is part of Q7.

### Q6 — How is performance scored? (`ObjectiveTerm` × `RewardConvention` × `Discounting`)
- **Cost terms**: `HoldingCost` · `BacklogCost` · `LostSalesPenalty` · `ProcurementCost` · `FixedOrderCost` (setup `K`) · `WasteCost` (outdating) · `SalvageCredit` · `EmergencyFulfillmentCost` · `Custom` (e.g. downtime, blending revenue, liquidation/return credits).
- **Convention**: `MinimizeCost` (most families) vs `MaximizeReward`/**profit** (joint-pricing, ameliorating-blending).
- **Horizon/discounting**: long-run **average** cost · **discounted** finite/infinite horizon (γ=0.99) · undiscounted finite horizon.
- **Tracked metrics**: `TotalCost` · `FillRate` · `CycleServiceLevel` · `AverageInventory` · `AverageWaste` · `AverageBacklog`.

### Q7 — What timing rules and feasibility constraints shape the system?
- **Shortage semantics** (`ShortageReaction`): `LostSales` · `Backorder` · `EmergencyFulfillment` · *hybrid/partial backorder* (OWMR β, divergent special delivery `P_w`).
- **Issuance rule** (`IssuanceRule`): `Fifo` · `Lifo` · `FixedPriority` · `Configurable` (perishable picks FIFO/LIFO per instance).
- **Lead time**: zero (joint-pricing, expedited source) · deterministic `L≥1` · per-echelon · stochastic/endogenous (axis exists, unused).
- **Order/feasibility constraints**: nonnegative integer order · order **cap** `q̄` · **fixed setup cost** `K` (lumpy ordering) · **FTL / truck-multiple** quantization (joint replenishment, total = multiple of `V`) · **capacity** caps on warehouse/retailer (divergent) · **shelf life** `m` (perishable) · **returnable cap** (procurement-removal).
- **Stage schedule** (`Stage`): `StartOfPeriod` → `AfterReceipts` → `AfterAction` → `AfterTransformations` → `AfterDemand` → `EndOfPeriod`, with the load-bearing **order-after-demand vs observe-then-receive-then-demand** convention recorded per family as a `TimingConstraint`.

---

## 3. Coverage map — how the 14 families answer the 7 questions

Compact answers; "primary new axis" is the one modeling dimension the family adds over its
predecessors. Difficulty is the provisional first-cut (the manifest field is being finalized
separately). Verification tier follows `VERIFICATION_LEDGER/README.md` (see §"Honest verification").

Legend — tier: **P** = verified_rerun vs a *peer-reviewed* number; **C** = verified_rerun vs
companion-code / closed-form / published-*action* (not a paper table); **F** = faithful_unverified
(repo-native self-consistency only, or no public number).

| Family | Q1 states | Q2 movement | Q3 random | Q4 choose | Q5 observe | Q6 score | Q7 timing/constraints | Primary new axis | Diff | Tier |
|---|---|---|---|---|---|---|---|---|---|---|
| **lost_sales / vanilla** (Zipkin 2008) | single-loc on-hand + pipeline | procure→ship→fulfill | i.i.d. Poisson/Geom/MMPP | ScalarOrder | FullState | hold + lost-sales, avg cost | **lost sales**, L≥1 | the base lost-sales primitive | easy | **P** |
| **lost_sales / fixed_order_cost** (Bijvank 2015) | + (same) | + (same) | i.i.d. | ScalarOrder | FullState | + **FixedOrderCost K** | lumpy (s,S) ordering | **fixed setup cost** | easy | **P** |
| **dual_sourcing** (Gijsbrechts 2022) | on-hand + regular pipeline (reduced) | two procurement channels (slow+fast) | i.i.d. U{0..4} | **DualSourceOrderPair** | ReducedState | hold + backlog + per-source cost, avg | backorder; `l_e=0`, `l_r≥1` | **second (expedited) supply channel** | medium | **P** |
| **multi_echelon / serial** (Clark–Scarf) | **SerialChain** echelon positions + pipelines | echelon-to-echelon shipment | i.i.d. | VectorOrder (echelon base-stock) | FullState | sum echelon hold + backlog | backorder; per-echelon L | **serial echelon coupling** (exact optimum) | medium | **P** |
| **multi_echelon / assembly** (Rosling 1989) | components → finished (assembly tree) | **Transformation** (assemble/merge) | i.i.d. | VectorOrder | FullState | hold + backlog | component sync; reduces to serial | **assembly / BOM convergence** | hard | **F** (equiv-only) |
| **multi_echelon / divergent_special_delivery** (Van Roy/Gijs) | **DivergentNetwork** wh + capacitated retailers | shipment + **special same-day delivery** | i.i.d. Normal-round | Allocation + warehouse order | FullState | wh+retailer hold + emergency + penalty, avg | **hybrid backlog/lost** (`P_w`); capacity caps | **divergent allocation + emergency channel** | hard | **C** (const base-stock ≤2%); A3C rows = debt |
| **multi_echelon / production_assembly_distribution_network** (Pirhooshyaran–Snyder) | **DirectedNetwork** raw+finished, single/assembly/distribution nodes | procure + **production** + ship across acyclic net | i.i.d. | **VectorOrder** (pairwise order-up-to) | FullState | per-node hold + pipeline hold + backlog | general acyclic protocol | **general directed supply network** | hard | **F** (single-node only verified) |
| **multi_echelon / general_backorder_fixed_cost** (Geevers / Kunnumkal–Topaloglu) | general network positions | ship across net | i.i.d. | VectorOrder | FullState | hold + backlog (**no fixed ordering cost despite carried family name**) | backorder; unit lead times | **general network node-base-stock targets** | hard | **P** (set1+KT); set2/3 = debt |
| **one_warehouse_multi_retailer** (Kaynov 2024) | wh + per-retailer positions + pipelines | wh-order + per-retailer shipment | i.i.d. per-retailer | wh order-up-to **+ Allocation** | FullState | wh hold + Σ(retailer hold + penalty), H=100 | **3 regimes**: lost / backorder / **partial-backorder β** | **scarce-warehouse allocation under regimes** | hard | **C** (2/14 rerun; rest table) |
| **perishable_inventory** (De Moor 2022 / Farrington 2025) | **AgeBucket** on-hand (one per shelf-life) + pipeline | + **Aging** + **Removal** (outdate) | i.i.d. rounded-Gamma | ScalarOrder | FullState | + **WasteCost**; disc. neg-cost | lost sales; **shelf life m**; FIFO/LIFO issuance | **inventory age profile / shelf life** | medium | **P** |
| **ameliorating_inventory** (Pahr & Grunow 2025) | **AgeBucket** by quality class + **price state** | Aging = quality **improves**; blending sale | i.i.d. demand + **stochastic purchase price** | ScalarOrder (purchase vol) | FullState | **profit** = revenue − purchase − hold, **avg** | blending; value rises with age | **value-increasing aging + stochastic price** | medium | **F** (LP bound = **C** vs companion) |
| **joint_replenishment** (Vanvuchelen 2020) | **JointMultiItem** per-item end inventory | per-item procure | i.i.d. per item | **VectorOrder** w/ shared total | FullState | major `M·K` + per-item `k_i` + hold + backlog, disc | **FTL truck-multiple** quantization | **shared major setup / multi-item coupling** | medium | **C** (published *action*) |
| **joint_pricing_inventory** (price-setting newsvendor) | single-loc on-hand (+ period) | procure (L=0) → fulfill | **price-elastic** demand | **order + discrete price** | FullState | **profit** = revenue − proc − hold − stockout, disc + salvage | lost sales; zero lead time | **price as a control (demand shaping)** | easy | **F** (no public number) |
| **nonstationary_lot_sizing** (Dehaybe 2024) | net inv + pipeline + **forecast window** | procure→arrive→fulfill | **nonstationary forecast-driven** | ScalarOrder | **ForecastAugmented** | setup K + proc + hold + penalty, total/mean | lost or backorder; rolling horizon | **nonstationarity via forecast signal** | medium | **C** (author CSVs) |
| **procurement_removal_inventory** (Maggiar–Sadighian 2017) | on-hand + **returnable pool** | procure (immediate) + **Removal/Return** | i.i.d. | **PurchaseAndRemoval** | FullState | proc + hold + shortage − return/liquidation credits, disc | lost sales; **returnable cap**; return-before-liquidate | **disposal / vendor-return channel** | easy | **F** (no public number) |
| **random_yield_inventory** (Yan 2026) | inv + pipeline (+ period) | procure with **stochastic yield** | **all-or-nothing batch yield** | ScalarOrder | FullState | proc + hold + backlog, disc | full backlog; L≥1 | **supply-side yield uncertainty** | easy | **F** (no public number) |
| **spare_parts_inventory** (Kranenburg 2006) | rotable on-hand + backlog + **procure & repair pipelines**, installed base | procure + **Repair** loop | **Failure** (binomial spares demand) | ScalarOrder (order-up) | FullState | proc + hold + **downtime** | order-after-demand; repairable loop | **repairable / failure-driven demand** | easy | **P** (Kranenburg module); env = **F** |
| **vendor_managed_inventory** (Sui–Gosavi–Lin 2010) | DC on-hand + retailer consignment + pipeline | DC→retailer **shipment** decision | i.i.d. retailer demand | ScalarOrder (shipment) | FullState | shipment + DC hold + retailer hold + penalty, disc | lost sales; 1-period ship lead | **consignment / who-decides shift (VMI)** | medium | **F** (no public number) |
| **decentralized_inventory_control** (Beer Game) | 4-stage serial, **per-stage local** inv/backlog/pipelines | per-stage upstream order | i.i.d. customer demand | order per stage (**multi-agent**) | **LocalState** | Σ stage (hold + backlog), undisc 36-wk | order-after-demand; information delay | **decentralized / local-information control** | hard | **C** (closed-form 204); env.rs = **F** |

---

## 4. Fundamentals coverage assessment

Read down the columns of §3:

**Well covered.**
- **Q1 topology** — every topology variant is present: SingleLocation, SerialChain, DivergentNetwork, DirectedNetwork, JointMultiItem.
- **Q1 state augmentation** — age profile (perishable, ameliorating), price state (joint-pricing, ameliorating), forecast window (nonstationary) are all instantiated.
- **Q4 action shapes** — Scalar, Vector, DualSourceOrderPair, Allocation, PurchaseAndRemoval, order+price are all exercised. (Only `Routing` is unused.)
- **Q6 scoring** — every `ObjectiveTerm` variant is used somewhere; both MinimizeCost and profit/MaximizeReward conventions appear; average, discounted, and undiscounted-finite horizons all appear.
- **Q7 shortage semantics** — lost sales, backorder, and both hybrids (partial-backorder β, special-delivery `P_w`) are covered; FIFO/LIFO issuance covered.

**Thin / single-instance.**
- **Q5 observation** — `LocalState` (decentralization) has exactly one family (Beer Game), and its trainable `env.rs` is unverified. `Delayed` observation is unused. Most families are `FullState`.
- **Q3 supply-side & non-demand randomness** — `Failure` (spare parts) and `Yield` (random-yield) each have one family; **`ReturnArrival` (closed-loop returns) is defined in the core enum but instantiated by NO family**; **`Disruption`/`TransitDelay` (stochastic/endogenous lead time) is defined but used by NO family** (all lead times are deterministic).
- **Q4 routing** — `Routing` action shape is unused (no transshipment/lateral-routing decision is a learned control; Kranenburg's lateral transshipment is analytical-only).
- **Q2 transformation/return** — `Return` and (production) `Transformation` are thin: production appears only inside PADN (unverified general protocol); `Return` only as a removal-credit in procurement-removal, not as a stochastic closed-loop inflow.

**Verification depth (orthogonal to coverage).** Only ~6 families carry a peer-reviewed `P` anchor;
the hard network/decentralized end of the space (PADN general protocol, divergent A3C rows, OWMR
12/14 rows, decentralized `env.rs`, assembly) is where faithfulness outruns verification. Coverage
of the *axis* and verification of the *number* are different debts — see VERIFICATION_LEDGER/README.md.

---

## 5. What could be added — candidate fundamental problems

Each candidate is justified by the existing 4-condition decision rule in
`fundamental_problem_families/README.md`: (a) classical family, not a narrow app variant; (b) at least one
citable RL/OR paper; (c) **adds exactly one new FlowNet-question answer** not yet covered; (d) admits
a verification anchor. Citations are given only where known; otherwise "anchor: TBD" (no fabrication).

### A. Deferred families already named in `fundamental_problem_families/README.md`
These are *implemented* now (ameliorating, procurement-removal, joint-pricing, vendor-managed) — the
"defer" note in that doc is stale and is updated by this pass. They are kept here only as the bridge
between the old expansion list and the live 14.

### B. Genuinely missing fundamentals (fill a THIN/EMPTY axis from §4)

1. **Substitution / assortment & demand-substitution.**
   - New answer: **Q3 demand** becomes *cross-product substitutable* (unmet demand for A spills to B) and **Q4** gains an *assortment/stocking-set* choice. No current family couples products through demand (joint-replenishment couples them only through a shared setup cost).
   - Anchor: Mahajan & van Ryzin (2001) stochastic substitution; RL anchor — Kaynov-style or *anchor: TBD*.

2. **Returns / remanufacturing & closed-loop inventory.**
   - New answer: **Q3** instantiates the unused `ReturnArrival` process (stochastic returns inflow) and **Q2** the unused `Return` + `Transformation` (remanufacture) edges; **Q1** adds a `Custom` *remanufacturable-core* node. This directly fills the biggest EMPTY enum slot.
   - Anchor: van der Laan & Salomon (1997) hybrid manufacturing/remanufacturing; DRL anchor *TBD*.

3. **Capacitated production–inventory (production smoothing).**
   - New answer: **Q7** adds a hard **production-capacity** constraint per period and **Q2** a real `WorkInProcess`→finished `Transformation`. Capacity caps exist on stock in divergent, but no family has a *production-rate* constraint with WIP.
   - Anchor: classic capacitated lot-sizing (Federgruen–Zipkin); DRL — Gijsbrechts/Boute roadmap context.

4. **Stochastic / endogenous lead time.**
   - New answer: **Q3** instantiates the unused `Disruption`/`TransitDelay` process — lead time itself is random (or order-quantity-dependent). Every current family uses deterministic `L`.
   - Anchor: Kaplan (1970) random lead time; supply-disruption RL — *TBD*.

5. **Advance demand information / pre-orders.**
   - New answer: **Q5** uses `ForecastAugmented` in a *committed-order* sense — the state carries known future demand (not just a forecast mean). Distinct from nonstationary-lot-sizing's noisy forecast.
   - Anchor: Gallego & Özer (2001) advance demand information.

6. **Multi-product capacity-shared newsvendor / shelf-space.**
   - New answer: **Q4** `VectorOrder` under a **shared resource (budget / shelf / warehouse) constraint** in **Q7** — a single-period or short-horizon allocation across products competing for one capacity. Joint-replenishment shares a *setup*, not a *capacity*.
   - Anchor: Hadley–Whitin constrained multi-item newsvendor; RL — *TBD*.

7. **Carbon / sustainability-constrained replenishment.**
   - New answer: **Q6** adds a `Custom` **emissions cost term** and **Q7** a per-period or cumulative **carbon cap** constraint — ordering trades off cost against an emissions budget.
   - Anchor: Hua, Cheng & Wang (2011) carbon-constrained EOQ; DRL — anchor: TBD.

8. **Transshipment / lateral pooling as a *learned* control.**
   - New answer: instantiates the unused **Q4 `Routing`** action — agents may laterally ship between same-echelon locations. Currently lateral transshipment exists only as Kranenburg's *analytical* module, not a trainable env.
   - Anchor: Kranenburg & van Houtum (2009) lateral transshipment (already in-repo, analytical); trainable-RL anchor — *TBD*.

**Prioritization (one-axis-at-a-time, per the decision rule).** The cleanest next additions are the
ones that turn an *already-defined-but-unused enum slot* into a real family — i.e. (4) stochastic
lead time and (2) returns/remanufacturing fill `Disruption`/`TransitDelay` and `ReturnArrival`/`Return`
respectively; (8) transshipment fills `Routing`. These maximize new axis coverage per unit of new
machinery and each has a classical OR anchor.

---

## 6. Cross-links

- **Expansion order & the 4-condition decision rule** → [`../literature/fundamental_problem_families/README.md`](../literature/fundamental_problem_families/README.md)
- **Honest per-problem verification provenance (3 tiers)** → [`VERIFICATION_LEDGER/README.md`](VERIFICATION_LEDGER/README.md)
- **Machine-readable instance index (difficulty, dimensions, flags)** → [`BENCHMARK_MANIFEST.json`](BENCHMARK_MANIFEST.json)
- **Suite index / master table** → [`README.md`](README.md)
- **The 7 questions in code** → `src/problems/core/flownet/question.rs`; the layers that answer them → `src/problems/core/{physical,stochastic,control,objective,timing}/`.

---

### Honest verification — reconciliation note

The coarse "STRICT-VERIFIED 6" framing is refined here to match `VERIFICATION_LEDGER/README.md` exactly:
- **Peer-reviewed verified_rerun (P)** = lost_sales/vanilla, lost_sales/fixed_order_cost,
  dual_sourcing, perishable_inventory, multi_echelon/serial, **multi_echelon/general_backorder_fixed_cost
  (set1+KT)**, **spare_parts_inventory/Kranenburg-module**. (The ledger's Group 1 includes the last two,
  which a "6-family" list omits — but note the *trainable* spare-parts env is only F, and gbk set2/3 is a debt.)
- **ameliorating_inventory is NOT peer-reviewed-verified**: only its perfect-info LP **bound** is
  verified_rerun *against companion code* (tier C); the trainable env is faithful_unverified (F).
- **multi_echelon/divergent_special_delivery is tier C, not F**: its constant-base-stock anchor
  re-runs Van Roy's 51.7/1302/1449 within 2% (51.77/1284.70/1437.96); only the A3C savings rows are a debt.
- **joint_replenishment (published action), nonstationary_lot_sizing (author CSVs), and
  decentralized_inventory_control (closed-form 204)** are tier C (companion/closed-form), not F.
