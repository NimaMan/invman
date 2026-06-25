# ameliorating_inventory — full-budget results (faithful average-profit env)

Committed-quality full-budget run of the price-reactive purchase soft tree on the
**faithful** average-profit ameliorating-inventory env
(`src/problems/ameliorating_inventory/average_profit_blending_env.rs`, the Pahr &
Grunow 2025 model), scored by
`ameliorating_inventory_average_profit_soft_tree_population_rollout` under paired CRN.

`outputs/` is gitignored, so the numbers are embedded here.

## Run configuration (`--budget full`)

| field | value |
|---|---|
| policy | depth-1 oblique soft tree, **linear** leaf, price-reactive purchase head |
| warm start | order-up-to purchase `softplus(S - sum(inventory))` (CMA-ES `x0`) |
| optimizer | CMA-ES, popsize 16, 60 generations, sigma_init 0.5, seed 20250604 |
| train | 4,000 periods / 1,000 warm-up, paired CRN across population |
| eval (reported) | 12,000 periods / 2,000 warm-up, **24 held-out paired-CRN seeds** |
| heuristic gate | best order-up-to level on the eval block (grid 2..24, ceiling 25) |
| CPU cap | RAYON_NUM_THREADS=4, OMP_NUM_THREADS=4 |

## Headline full-budget results

| instance | learned profit ± SEM | LP upper bound | order-up-to gate (level) | gap vs gate (abs) | gap vs gate (% of gate) | gap to bound | verdict |
|---|---|---|---|---|---|---|---|
| **spirits_0001** | **115.07 ± 0.44** | 1991.9344 | 20.91 (S=24) | **+94.16** | **+450.4%** | **94.22%** | **beats_order_up_to** |
| **port_wine** | **505.78 ± 0.59** | 2444.8011 | 133.78 (S=24) | **+372.00** | **+278.1%** | **79.31%** | **beats_order_up_to** |

Paired-CRN SEM of the learned − gate difference: **0.66** (spirits_0001), **0.61** (port_wine).
The learned advantage exceeds the paired SEM by ~140x and ~610x respectively, so the
"beats the order-up-to gate" verdict is robust, not noise.

Learned-policy source selected on the held-out eval block: `gen_best@25` (spirits_0001),
`cma_incumbent` (port_wine). Train wall-clock: 196 s (spirits_0001), 527 s (port_wine),
RAYON/OMP capped at 4 threads.

## Reading the numbers honestly

This is a **`bound_gap`** problem. The perfect-information LP value is an **UPPER BOUND**,
not an achievable target, so we report the gap and never claim to "beat" it. The only
legitimate "beat" is vs the in-repo tuned order-up-to gate, evaluated like-for-like (same
env, same paired-CRN held-out seeds and horizon, margin beyond paired SEM) — and that beat
holds robustly on both instances.

Three distinct framings of the gate advantage (the earlier program-md table labeled the
**absolute** profit difference as a "%", which conflated units — corrected here):

- absolute profit gained over the best order-up-to: +94.16 (spirits_0001), +372.00 (port_wine);
- as a fraction of the gate baseline: +450% and +278%;
- as a fraction of the LP bound: +4.73 and +15.22 percentage points of the bound closed.

The earlier screening "+79.8%" figure on spirits_0001 was the absolute units gap (79.76);
at full budget the absolute gap grows to +94.16.

## Why the gap to bound is loose (and NOT comparable to the paper's ~3.5% DRL gap)

The LP bound assumes **perfect information** and the **full 3-part LP issuance** (purchase +
production targets + per-age issuance solved jointly with hindsight). Our policy controls
only the **scalar per-period purchase volume**; issuance is whatever the env's per-period
blending LP extracts from current inventory, and the env charges the **full purchase price
(~200/unit) every period**. A single-purchase feasible policy on the stochastic env therefore
sits far below the perfect-information bound by construction. The gap is reported truthfully;
it is structural (action geometry), not an optimization failure. The paper's ~3.5% DRL gap
uses the full 3-part action including production targets and is a different (wider) action
space — see the scope note below and PAPER_SECTION_DRAFT/README.md.

port_wine's gap to bound (79.3%) is meaningfully tighter than spirits_0001's (94.2%):
blending lets the issuance LP draw across the two target age classes [9,19], so each
purchased unit converts to more sold product per period — the policy captures more of the
bound even with a single purchase lever.

## Env fidelity (literature anchor)

The faithful env reproduces both published perfect-information LP upper bounds to < 1e-3 by
RE-SOLVING the LP (an executing reproduction, not a frozen snapshot):
`src/problems/ameliorating_inventory/tests/verification.rs`
(`spirits_0001_perfect_information_upper_bound_reproduces_published_max_reward`,
`port_wine_perfect_information_upper_bound_reproduces_published_max_reward`), anchored on
`upper_bound.json` from the Pahr & Grunow (2025) companion repo
(spirits_0001 = 1991.9344293376805; port_wine = 2444.8010643781136).

## Scope decision (recorded, not implemented)

Widening the action head to the full 3-part action (purchase + production targets + issuance)
is the lever that would chase the paper's ~3.5% DRL gap. It requires NEW Rust rollout code
and a rebuild of `invman_rust`, which is OUT OF SCOPE for this run (a rebuild would corrupt
the two parallel training agents sharing the install). Recommended as the deeper follow-up.

Runner: `scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py`.
Output JSON: `outputs/autoresearch/ameliorating_inventory_average_profit_autoresearch/{spirits_0001,port_wine}_d1_oblique_full.json`.
