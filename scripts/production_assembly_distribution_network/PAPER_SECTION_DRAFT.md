# Paper subsection draft (HONEST) — learned control on a faithful production/assembly/distribution network

> Draft text for a paper subsection. NOT committed into `paper/*.tex`. The orchestrator decides
> whether/how to fold this into the manuscript. Numbers refreshed 2026-06-04 against install
> commit 2bb8df8; full ledger in
> `scripts/production_assembly_distribution_network/RESULTS_case3_learned_vs_own_best_heuristic.md`.

## Framing rule for this subsection (do not violate)

This environment is **faithful but not number-anchored**. The result is a **learned-policy vs the
environment's own best heuristic** comparison on a research environment. It is **NOT** a beat against
a published benchmark, and it must never be written as one. The single published number this family
reproduces (the single-node newsvendor cost 127.11) certifies the environment's DYNAMICS, not the
case3 result.

---

## Subsection: A learned tree policy on a faithful supply-network MDP

### Environment

We use the finite-horizon stochastic supply-network MDP of Pirhooshyaran & Snyder (2021,
arXiv:2006.05608): a directed acyclic network of nodes carrying raw-material and finished-goods
inventories, pairwise order-up-to decisions on supply relations, "process all raw on arrival"
production, proportional downstream allocation with backorder carry-over, and an order-after-demand
sequence (their equations 1-13; cost equation 3 charges holding on raw, finished, and in-transit
inventory). Our implementation is verified equation-by-equation against the paper, and a
worked-transition fixture reproduces the per-period cost by hand.

We make a deliberate honesty commitment about this environment's status. **It is faithful but not
literature-verified.** The only published quantity it reproduces is the single-node newsvendor cost
of the paper's Table 1: for the (μ=100, σ=10, h=10, p=30, L=1, T=2) row, the environment's exact
dynamic program at the published order-up-to level returns ≈127.10 against the published 127.11
(<1% relative gap; residual is integer demand/level discretization). This certifies the
**dynamics** are correct. It does **not** certify any multi-node optimum: there is no published
optimum for the multi-node version of *this* MDP, so every multi-node reference instance carries
`literature_verified = false`.

In particular, the textbook serial optimum 47.65 (Snyder & Shen Example 6.1, also Pirhooshyaran
Table 3 case 3) is **structurally unreachable** by this environment as a target. That optimum is an
*echelon* base-stock cost; the paper's pairwise policy (their equation 5) controls the *local*
raw-material inventory position of each supply relation, which excludes finished goods. Because each
node processes all raw material on arrival, over-produced finished goods accumulate invisibly to the
local position; driving the environment's local pairwise policy with the analytically derived levels
yields costs far above the environment's own best (see Section "Why analytical levels do not
transfer" below). The serial optimum's verified home is a separate serial-echelon family, not this
network environment. We therefore do **not** use 47.65 (or any published number) as the comparator
here.

### Instance and protocol

We study the paper's serial case 3 (`pirhooshyaran2021_serial_case3`): a 3-node serial chain
0 → 1 → 2 with node 0 the sole source, external demand N(5,1) at node 2 only, horizon T = 10, lead
times (external→0, 0→1, 1→2) = (2, 1, 1), local holding [2, 4, 7], backorder cost 37.12 at node 2.
The objective is undiscounted average per-period cost. We evaluate under common random numbers
(CRN): a search demand-path block and a disjoint held-out block (4000 paths), with the **same**
held-out block scoring every policy (paired / variance-reduced).

**Comparator (the environment's own best heuristic, a research baseline, not an optimum).** The
strongest heuristic native to this environment is the pairwise base-stock policy. We grid-search its
per-relation order-up-to levels on the search block and re-score the argmin on the held-out block.
The argmin is order-up-to [8, 7, 9] (relations = edge 0→1, edge 1→2, external→node 0), with held-out
cost **60.24 per period**. This is the keep/discard gate. We state explicitly that 60.24 is the
environment's own best heuristic, *not* an optimum.

### Policy

The learned policy emits a **direct order quantity per supply relation** (a `vector_quantity` action
of dimension 3) via a depth-2 oblique soft decision tree with **linear leaves**, clipped to the
physical action box. The design choice we test is the *leaf class*, not the optimizer budget: a
constant leaf can emit only a fixed order rate and cannot express order-up-to behavior, whereas a
linear leaf maps the policy-state features — which include per-relation raw inventory, in-transit
pipeline, finished inventory, and backlog — to the per-relation order, so it can express
inventory-position feedback, and oblique splits let it switch regime by inventory state. We train the
tree with CMA-ES (popsize 24, 60 generations, paired CRN), warm-started at the steady-state flow rate
(order the demand mean ≈5 per relation per period), the simplest reproducible known-good point; the
search then refines outward.

### Result

The learned tree beats the environment's own best pairwise base-stock on the held-out block, robustly
and reproducibly:

| Policy / config | Held-out cost ± SEM (per period) | vs gate (60.24) |
|---|---|---|
| Best pairwise base-stock (gate, OUL [8,7,9]) | 60.24 ± 0.30 | — |
| Flow warm-start (gen 0) | 70.85 ± 0.61 | +17.6% |
| Learned soft tree, depth 2, seed 123 | **57.25 ± 0.22** | **−4.96%** |
| Learned soft tree, depth 2, seed 321 | **54.96 ± 0.23** | **−8.77%** |
| Learned soft tree, depth 3, seed 123 | **57.85 ± 0.25** | **−3.97%** |

The improvement is **−5% on the headline config** (and up to −9% across CMA seeds), outside the
held-out standard error by ≥9 SEM, and reproduced across two CMA seeds and two tree depths. The
mechanism is a strictly richer control class on the **same** action relations: the pairwise
base-stock policy reacts only to the local raw-material position, while the learned linear-leaf tree
additionally reads finished inventory, internal/external backlog, and inbound pipeline, and switches
behavior by inventory regime. Constant-leaf trees stay at the flow regime and lose to the gate,
confirming the leaf class is the lever.

**Honest scope.** This shows the learned policy beats the *environment's own best heuristic* on a
faithful but non-literature-verified network MDP. It does **not** reproduce or beat any published
cost. We report it as evidence that action design (the leaf class), not optimizer capacity, governs
whether a black-box search recovers structured-control performance — consistent with the same
finding on the lost-sales, one-warehouse-multi-retailer, and multi-echelon environments elsewhere in
this work.

---

## Path-B feasibility note: could this environment become literature-verified?

**Question.** Could Pirhooshyaran's exact order-up-to → inventory-position simulation protocol be
recovered so this environment reproduces a *published* serial/network cost (e.g. Table 3 case 3),
which would upgrade it to `literature_verified = true`?

**Verdict: feasible but nontrivial; the lever is the policy/position convention, NOT the environment
dynamics. We did not attempt it (env is frozen).**

Findings (grounded with the existing Python bindings — no rebuild):

1. **Dynamics are not the problem.** The environment already reproduces the single-node newsvendor
   cost (127.11 → ~127.10) by simulating its own dynamics, and an in-crate worked-transition fixture
   matches by hand. The paper sets production time to zero and so does the environment; an impulse
   order placed at the source reaches finished goods after exactly its shipment lead time, so the
   effective serial lead time is 2+1+1 = 4, matching Clark-Scarf. Holding-on-in-transit is faithful
   to cost equation 3. A destructive dynamics rewrite is neither needed nor warranted.

2. **The gap is a local-vs-echelon policy/position mismatch.** The environment's pairwise policy
   (equation 5) targets the *local* raw-material inventory position
   (`raw_inventory_by_relation − total_current_demand + in_transit + predecessor_backlog`), which
   excludes finished goods. Feeding analytically derived levels into this local policy does not
   transfer (held-out costs, this instance):
   - carried analytical pairwise OUL [10.69, 5.53, 6.49] → **95.70 ± 0.72**;
   - exact serial-family *local* levels mapped per relation [5.41, 6.70, 10.30] → **113.96 ± 0.89**;
   - the environment's own grid argmin [8, 7, 9] → **60.24 ± 0.30**.
   The analytical levels are 1.6–1.9× worse than the environment's own grid argmin, confirming the
   level-interpretation mismatch quantitatively.

3. **"47.65" is convention-specific, even within the verified serial family.** The repo's verified
   serial-echelon exact solver (Clark-Scarf, Normal demand), run on the case3 parameterization
   (echelon holding [2,2,3], leads [1,1,2], penalty 37.12, N(5,1)), returns an infinite-horizon
   optimal cost of **59.05**, not 47.65. The paper's 47.65 is its own finite-horizon (T=10)
   simulation under a specific warm-start and position convention. Notably, the environment's own
   best pairwise base-stock (60.24) is already within ~2% of the verified infinite-horizon serial
   optimum (59.05), and the learned policy (≈55–58) goes below it — expected, because case3 is a
   finite-horizon problem (T=10) with a specific initial inventory, not the stationary system.

4. **What it would take.** Two viable routes, neither requiring a dynamics change:
   - (a) Recover Pirhooshyaran's exact OUL → inventory-position simulation protocol for their
     pairwise base-stock run (the position convention and warm-start that make their Table-3
     simulation yield 47.65), expose the echelon-position policy already present in-crate
     (`serial_echelon_simulation.rs::echelon_base_stock_requests`, currently NOT bound to Python),
     add the published cost as a verification target, and assert env simulation reproduces it within
     ~2%. The blocker is that the OUL levels alone are insufficient — the position definition and
     warm-start matter, and the paper does not fully specify them — so this requires either author
     contact or careful reverse-engineering, and a rebuild to expose the binding.
   - (b) Compute the correct *env-native* local/echelon base-stock levels for THIS environment (by
     direct search over per-relation levels) and record both the levels and the resulting simulated
     cost as a self-consistent anchor, explicitly labelled an env-native optimum rather than a
     paper-published number. This does **not** make the env `literature_verified` (no published
     number is reproduced) but it removes the "60.24 is just a grid argmin" caveat.

Until env simulation re-derives a *published* cost within ~2%, the environment stays
`literature_verified = false` and the case3 result remains a research comparison against the
environment's own best heuristic. We recommend route (a) as a future task (requires a rebuild to bind
the echelon-position policy) and do not modify the frozen environment here.
