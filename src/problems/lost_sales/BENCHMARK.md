# lost_sales — benchmark card

**One-line MDP:** state = on-hand inventory + outstanding pipeline `(I_t, q_{t-L+1},…,q_{t-1}) ∈ Z≥0^L`; action = integer order `q_t ∈ {0,…,q̄}`; one-period cost = `h·(I_t−D_t)^+ + p·(D_t−I_t)^+` (vanilla) plus `K·1{q_t>0}` (fixed-cost variant); objective = minimize long-run average expected cost (T→∞), with unmet demand **lost** (not backordered).

**Status:** **verified_rerun** (two sub-families). fixed_order_cost = genuine in-repo EXACT solver matches Bijvank 2015 Table 1; vanilla = 3 heuristic rows re-run against Zipkin 2008 Table 3(a) (the optimum 4.73 is a *carried* Zipkin DP value, not recomputed in-repo). **Paper:** §"Lost sales and fixed-cost lost sales" of learning_inventory_control_policies_es.tex (line 503).

## Problem formulation

Single-item, periodic-review lost-sales control; integer demand `D_t ~ D` stationary; integer orders; fixed lead time `L ≥ 1`.

- **Timing of a period t:** (i) the order `q_{t-L}` placed L periods earlier arrives and is folded into leftover on-hand, giving on-hand `x_t = I_t`; (ii) the controller observes the state and places `q_t ∈ {0,…,q̄}`, to arrive at the start of `t+L`; (iii) demand `D_t` is realized and served from on-hand only — sales are capped at `x_t` and **unmet demand is lost**; (iv) holding + lost-sales (+ setup) costs are charged on the period outcome.
- **State:** `S_t = (I_t, q_{t-L+1},…,q_{t-1}) ∈ Z≥0^L` — first coordinate is on-hand after arrival; remaining `L−1` are the outstanding pipeline. State space grows exponentially in L (the problem is notoriously hard; no tractable exact VI for the vanilla pipeline).
- **Transition:** `I_t = (I_{t-1} − D_{t-1})^+ + q_{t-L}`; pipeline drops the just-arrived order and appends `q_t`: `S_{t+1} = ((I_t − D_t)^+ + q_{t-L+1}, q_{t-L+2},…,q_{t-1}, q_t)`. Unsatisfied demand `(D_{t-1} − I_{t-1})^+` is never carried forward.
- **One-period cost (vanilla):** `c_t = h·(I_t − D_t)^+ + p·(D_t − I_t)^+`. No fixed cost.
- **One-period cost (fixed-cost variant):** `c_t^K = c_t + K·1{q_t>0}` — identical state/dynamics/lost-sales rule; the setup term makes order/no-order an explicit MDP decision (no longer optimal to order every period).
- **Objective:** `min_θ C̄(θ)`, `C̄(θ) = lim_{T→∞} (1/T) Σ E[c_t]` (infinite-horizon average cost); estimated in practice by a long simulation rollout with warm-up discard over horizon ~10^6 with 10 consecutive seeds.

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `vanilla_l4_p4_poisson5` (alias `lit_poisson_p4_l4`) | vanilla; Poisson; canonical anchor | μ=5, L=4, h=1, p=4 (Zipkin 2008 Table 3a) | **true** |
| `lit_poisson_p19_l4` | vanilla; Poisson; high penalty | L=4, p=19 | absent — reference_costs.rs has no per-row flag; source=literature, carries published optimal 8.89 |
| `lit_poisson_p4_l6 / l8 / l10` | vanilla; Poisson; deep pipeline | L=6/8/10 | absent — source=literature(+computed); heuristic cells are full-precision repo-computed (e.g. myopic1=5.4140775) |
| `lit_geometric_p4_l4 .. lit_geometric_p19_l10` (8 rows) | vanilla; Geometric; high CV | L=4–10, p∈{4,19} | absent — source=literature+computed; M1/SVBS repo-computed |
| `lit_mmpp2_pos_* / lit_mmpp2_neg_*` (16 rows) | vanilla; MMPP2; autocorrelated demand | pos: p00=p11=0.9 / neg: p00=p11=0.1; L=4–10; p∈{4,19} | absent — source=computed; order qty on stationary marginal, cost on true MMPP2; no capped_base_stock |
| `bijvank2015_table1_l2_p14_k5` | fixed_order_cost; Poisson; **exact-solvable** anchor | μ=5, L=2, h=1, p=14, K=5 (Bijvank 2015 Table 1) | **true** |
| fixed-cost full grid `lost_sales_style_full_grid_mu5` (80 instances) | fixed_order_cost; {Poisson, Geometric, MMPP2+/−}; L=2/4/6/8/10; p∈{4,19}; K∈{5,25} | μ=5 surface | absent — README: larger grids not yet literature-verified |

(Reported paper surface: vanilla = 3 demand families × L∈{4,6,8,10} × p∈{4,19} = 24 instances; fixed-cost adds K∈{5,25} → 48 instances. The reference_costs grid is a 33-instance superset.)

## Baselines

- **Heuristics (vanilla):** Myopic-1 (Zipkin "Myopic"), Myopic-2, Standard Vector Base Stock (SVBS, Morton 1969/71), Better/Capped Vector Base-Stock (Zipkin "Better VBS" 4.80, corroborated by Xin 2021). Costs from a live env + heuristic rollout (horizon 10^5, seed 123 for the published anchor; horizon 10^6 / 10 seeds for the paper surface).
- **Heuristics (fixed-cost):** `(s,S)`, `(s,nQ)`, modified `(s,S,q)` — exact evaluation via bounded-DP for the published instance; literature-reported parameters where available.
- **Exact / optimal:**
  - fixed_order_cost: `exact_value_iteration.rs` — average-cost bounded DP / relative value iteration over the lost-sales pipeline (exact within an inventory-position cap; cap≥24 matches Bijvank tightly). This is a **genuine in-repo exact solver**.
  - vanilla: **NO in-repo exact solver.** The optima 4.73 / 8.89 / 10.61 / 22.95 / etc. are **carried published Zipkin DP values, not recomputed in-repo.**
- **Published comparators (CONTEXT):** No published DRL row exists in the lost-sales tables. Gijsbrechts et al. 2022 (Management Science 68(3)) is carried as **context only** — it restates Zipkin's μ=5 instance but provides no comparator row used here.

## Verification

- **fixed_order_cost (Bijvank 2015 Table 1, L=2 p=14 K=5 Poisson(5)):** Published optimal 11.46, (s,S) 11.62, (s,nQ) 11.56, modified (s,S,q) 11.50. **Re-run reproduced (cap=24, exact DP):** optimal **11.4631** (gap +0.0031, first_action=8), (s,S) **11.6181**, (s,nQ) **11.5552**, modified **11.4974** — all gaps <0.005. Via `ir.lost_sales_fixed_order_cost_exact_literature_summary('bijvank2015_table1_l2_p14_k5',24)`. **Verdict: verified_rerun (genuine EXACT solver match).**
- **vanilla (Zipkin 2008 Table 3a, L=4 Poisson(5) h=1 p=4):** Published Myopic 5.06, Myopic-2 4.82, SVBS 5.83 (optimal 4.73). **Re-run reproduced (horizon=100k, seed=123):** myopic1 **5.0569**, myopic2 **4.8208**, svbs **5.8153** — all within ~0.015. Via `ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995)`. **Verdict: verified_rerun for the 3 heuristic rows.**
- **Debts / caveats (state plainly):**
  - The vanilla **optimum 4.73 is a carried Zipkin value, not recomputed** — there is no in-repo exact VI for the vanilla pipeline; only the 3 heuristic rows were re-run.
  - **Only `vanilla_l4_p4_poisson5` carries true Zipkin numbers.** The rest of the vanilla grid (Geometric, MMPP2, and the deep-pipeline Poisson rows) is **repo-computed**, not literature-transcribed.
  - The fixed-cost **80-instance full grid** beyond the Bijvank anchor is **not literature-verified** (larger grids; per README).

## Results (learned policy)

- **Seed-robust (at_risk=false):** Canonical vanilla L4-Poisson p4, depth-2 soft tree (Tree-2) reaches **learned 4.7537**, within eval noise of the published optimum **4.73**. The manifest marks this row **multi_seed_mean_std** (the only seed-robust learned result for this system).
- **Single-seed, NOT yet seed-robust (at_risk=true, label honestly):**
  - Vanilla surface: "learned policies best in **22/24** reported instances; on L4-Poisson4 depth-2 linear-leaf soft tree 4.7537 vs myopic2 4.8186 (−1.20%)". Manifest seed_reporting = **single_seed** — the per-instance surface table (Table results-vanilla-lost-sales) is single-seed; do NOT treat the 22/24 sweep as seed-robust.
  - Fixed-cost surface (48 instances): "learned competitive/winning vs (s,S)/(s,nQ)/(s,S,q); canonical L4 Poisson K5 p4 learned ~8.73 vs heuristic ~9.20". Manifest seed_reporting = **single_seed**.
- Paper Table (single-seed surface, illustrative anchors): vanilla L4-Pois p4 — M1 5.065, M2 4.819, SVBS 5.836, Tree-1 4.749, Tree-2 4.750 (Optimal 4.730). The two vanilla instances still won by classical baselines are the high-penalty MMPP2+ cases at L=8 (SVBS) and L=10 (Myopic-2).

## Reproduce

```bash
# (1) vanilla heuristic re-run vs Zipkin 2008 Table 3a (canonical anchor)
python -c "import invman_rust as ir; print(ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995))"

# (2) fixed_order_cost exact DP re-run vs Bijvank 2015 Table 1
python -c "import invman_rust as ir; import json; print(json.dumps(ir.lost_sales_fixed_order_cost_exact_literature_summary('bijvank2015_table1_l2_p14_k5',24), default=str))"

# (3) dump every reference instance's source + carried costs
python -c "import invman_rust as ir; [print(n, ir.lost_sales_reference_costs(n)['source'], ir.lost_sales_reference_costs(n)['costs']) for n in ir.lost_sales_reference_instance_names()]"

# (4) expand the 80-instance fixed-cost grid
python -c "import invman_rust as ir; g=ir.lost_sales_fixed_order_cost_expand_experiment_grid('lost_sales_style_full_grid_mu5'); print(len(g), g[0]['name'], g[-1]['name'])"

# (5) reference-instance validation harness
python /home/nima/code/ml/invman/scripts/lost_sales/validate_reference_instance.py --num_seeds 3

# (6) full learned-policy benchmark suite
python /home/nima/code/ml/invman/scripts/lost_sales/benchmark_full_suite.py --seed 42 --eval_seeds 10
```

## Pointers & caveats

- **code:** `src/problems/lost_sales/` — `mod.rs`; vanilla: `vanilla/env.rs`, `vanilla/heuristics/`, `vanilla/reference_costs.rs`, `vanilla/literature/references.rs`, `vanilla/rollout.rs`, `vanilla/flownet/`; fixed-cost: `fixed_order_cost/exact_value_iteration.rs`, `fixed_order_cost/heuristics.rs`, `fixed_order_cost/literature/references.rs`, `fixed_order_cost/verification/tests.rs`; demand: `demand/iid.rs`, `demand/markov_modulated.rs`.
- **scripts:** `scripts/lost_sales/` (`validate_reference_instance.py`, `benchmark_full_suite.py`, `generate_rust_reference_costs.py`, `autoresearch_lost_sales.py`).
- **autoresearch:** `autoresearch/program_lost_sales.md` (benchmark fixed to the trusted vanilla L4/p4/Poisson(5)/h=1 instance; harness files are pinned, do not modify).
- **Honest caveats:**
  - Vanilla optimum 4.73 (and the other carried optima) is a **published Zipkin DP value, NOT an in-repo recomputation** — only the 3 heuristic rows are re-run; only `vanilla_l4_p4_poisson5` carries true Zipkin numbers, the rest of the grid is repo-computed.
  - Gijsbrechts 2022 DRL is **context only** (no published DRL comparator row in these tables; never a "beats" claim).
  - The 22/24 vanilla sweep and the 48-instance fixed-cost sweep are **single-seed (at_risk), NOT seed-robust**; only the canonical vanilla L4-Poisson p4 Tree-2 = 4.7537-within-noise-of-4.73 result is multi-seed.
  - Demand convention: Poisson/Geometric are mean-preserving (E[D]=5); MMPP2 rows are computed (order on stationary marginal, cost on true MMPP2), not literature-transcribed.
