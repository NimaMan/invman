# random_yield_inventory

Rust-first problem home for `random_yield_inventory`.

## Formulation

Single-item, periodic-review inventory with **all-or-nothing supply yield** and a positive
deterministic lead time, finite horizon, discounted cost, full backlogging.

State (`env.rs`): `(period, inventory_level, pipeline_orders[L])`. Action: the order quantity placed
this period. Order of events per period (`env.rs::step_state`):

1. the oldest pipeline order arrives in full with probability `p` (success) or not at all with
   probability `1-p` (all-or-nothing yield): `realized_arrival = pipeline[0]` if success else `0`;
2. demand is realized; `ending = inventory + realized_arrival - demand`;
3. period cost `= procurement_cost * order + holding_cost * ending^+ + shortage_cost * ending^-`;
4. the pipeline shifts and the new order is appended: `next_pipeline = pipeline[1..] ++ [order]`.

An order placed now therefore arrives after exactly `L` periods (clean lead-time-`L` pipeline).

Code lives under `rust/src/problems/random_yield_inventory/`.

Literature and verification anchors:

- `literature/references.rs` — cited papers, reference instances, literature benchmark families
- `finite_horizon_dp.rs` — exact reduced finite-horizon DP (optimal + heuristic evaluation)
- `heuristics/` — LIR (linear inflation) and WNH (weighted newsvendor)
- `verification/tests.rs` — executable implementation-correctness assertions
- `literature/`, `practical/`, `experiments/`, `verification/` — README scope notes

## Verification status: SELF-CONSISTENT-ONLY (not literature-verified)

This is an honest, evidence-backed classification (verified during the 2026-05 review). All three
cited papers were independently confirmed to be **real and correctly cited** (Crossref, DBLP, RePEc,
publisher / open working-paper PDFs): Yan et al. (2026) C&OR 186:107305, doi:10.1016/j.cor.2025.107305;
Chen et al. (2018) IEEE SOLI pp. 180–184, doi:10.1109/SOLI.2018.8476751; Inderfurth & Kiesmüller (2015)
EJOR 245(1):109–120, doi:10.1016/j.ejor.2015.03.006. The status below concerns *verifiability against a
published number*, not the correctness of the citations.

- The MDP transition + cost (`env.rs`, `finite_horizon_dp.rs`) **faithfully match the structure** of
  the cited Yan et al. (2026) all-or-nothing / positive-lead-time / discounted / backlog model. This is
  a structural (model-fidelity) match, **not** a reproduced literature number.
- The exact DP (`finite_horizon_dp.rs`) is **implementation-correct**: it was re-derived from scratch
  in an independent Python DP of the same MDP and reproduces the optimal cost
  `40.0598976099` and first action `4` on `VERIFICATION_PROBLEM_INSTANCE` to full precision. Lifting
  the DP action cap from 8 to 20 changes the optimum only at the 5th significant figure
  (`40.0598742583`), so the carried cap is effectively non-binding for the optimal policy. This is a
  **repo-native self-consistency** check against the repo's own exact solver.
- BUT there is **no public per-instance benchmark number** to assert against: Yan et al. (2026) and
  Chen et al. (2018) are paywalled and expose no reusable table; Inderfurth & Kiesmüller (2015)
  publish numbers only for a **different yield model** (per-unit binomial / stochastically
  proportional, infinite-horizon average cost), not this finite-horizon all-or-nothing batch model
  (this was confirmed by reading the open working-paper PDF in the 2026-05 audit).

So the verifier is a repo-native, exact-solver self-consistency check — it confirms the code is
correct, not that it reproduces a literature number. The accurate taxonomy status is therefore
**self-consistent-only**: validated against the repo's own exact solver, with no public anchor. This is
strictly weaker than the finished problems (e.g. `lost_sales_fixed_order_cost` reproduces a Bijvank
2015 Table 1 number; `dual_sourcing` reproduces Gijsbrechts 2022 Figure 9 gap labels) —
random_yield_inventory has no equivalent public anchor, so it correctly does not claim that status.

### Open fidelity question (root cause of the remaining gap)

The **WNH (weighted newsvendor) order rule** in `heuristics/weighted_newsvendor.rs` computes the
yield-weighted expected gap `E_pipeline_scenarios E_demand[(S - projected_inventory)^+]` but does
**not** multiply that gap by the reciprocal of the mean yield rate `1/p`. Two independent secondary
descriptions of the all-or-nothing heuristics (the Yan 2026 abstract record and the Chen 2018 record)
state the order is "the gap ... multiplied by the reciprocal of the mean yield rate", which would
inflate the WNH order by `1/p` (as the LIR already does in `heuristics/linear_inflation.rs`). The
exact published WNH formula is paywalled and could not be recovered, so this was **not changed** —
inflating the WNH would only push its already-overshooting order (8 vs optimal 4 on the verification
instance) further up. This is recorded as a precise next step rather than a guess.

The LIR is faithful: `q = (1/p) * (S - X)^+` with inventory position `X = inv + p * sum(pipeline)` and
order-up-to target `S = Poisson((L+1) * mean).invcdf(b/(h+b))` (textbook protection interval `L+1`).

## Benchmark (2026-05)

Run with `scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py`.

**Exact-DP slice** (`VERIFICATION_PROBLEM_INSTANCE`, capped, discrete demand, L=2, implementation-
verified optimum):

| Policy | Discounted Cost | First Action | Gap to Optimal |
| --- | ---: | ---: | ---: |
| `exact_optimal_dp` | 40.0599 | 4 | 0.0000 |
| `linear_inflation` (LIR) | 47.7138 | 4 | 7.6539 |
| `weighted_newsvendor` (WNH) | 60.3936 | 8 | 20.3337 |

LIR is the stronger heuristic here (19.1% above optimum); WNH overshoots (50.8% above optimum).

**Simulation slice** (`PRIMARY_REFERENCE_INSTANCE` = `yan2026_style_lt2_p075_discounted`, Poisson
demand, horizon 12, 2000 held-out evaluation seeds, uncapped env). The soft-tree was CMA-ES-trained
(depth 3, 600 episodes, population 32) on disjoint training seeds:

| Policy | Mean Discounted Cost | Std | Gap to Best |
| --- | ---: | ---: | ---: |
| `soft_tree(d=3, linear)` | 196.661 | 114.290 | 0.000 |
| `linear_inflation` (LIR) | 203.619 | 123.769 | 6.959 |
| `weighted_newsvendor` (WNH) | 222.436 | 66.918 | 25.776 |

The learned soft-tree beats LIR by 3.4% and WNH by 11.6% in mean discounted cost out-of-sample (note
WNH trades higher mean cost for markedly lower variance).

## State interface

- `env.rs` exposes raw state quantities only.
- The soft-tree benchmark keeps its derived/normalized feature map in `rollout.rs`.
- Any normalization or expectation-based encoding stays outside the environment layer.
