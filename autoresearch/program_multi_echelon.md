# Multi-echelon policy design (Van Roy 1997 / Gijsbrechts 2022 one-warehouse, K-retailer)

This is the policy-design program for the `divergent_special_delivery` problem: one capacitated
warehouse replenishing `K` identical capacitated retailers, with same-day **special delivery**
(unmet retailer demand is expedited from the warehouse with probability `Pw`, otherwise lost) — a
hybrid backlog/lost-sales divergent two-echelon system.

## Design philosophy: design the policy for the problem, not to match a paper

**Our training is separate from Gijsbrechts' training.** Gijs's action grid, their A3C network,
their feature engineering, and their hyper-parameter tuning are *their* design decisions for *their*
method. We do not copy them. We design a policy that is appropriate for the **problem** (the MDP
defined by the env), and we use the literature numbers purely as a **benchmark to beat**:

- We **use the published numbers** (constant base-stock costs, NDP/A3C costs and relative savings) as
  the comparison targets.
- If our problem-appropriate policy **beats** the published results, we **report that** (it is a
  legitimate result, not a reproduction failure).
- We do **not** constrain our action space, features, or training to reproduce the paper's choices.

The only thing that must be faithful is the **environment** (the MDP transition + cost). That is
verified — see `rust/src/problems/multi_echelon/divergent_special_delivery/` (gijs_2022 mode:
pre-shipment warehouse order Eq. (2), end-of-period holding; exact-DP + worked-transition tests pass)
and its `literature/README.md` (Van Roy constant base-stock costs reproduce within ~1% on the
`van_roy_1997` reproduction instances).

## Two instance families (see `references.rs`)

- **Paper-faithful search targets** (`gijs_2022` dynamics, Table-3 demand): `gijsbrechts2022_setting1`
  (lw=2, lr=2, μ=5, σ=14, K=10), `gijsbrechts2022_setting2` (lw=5, lr=3, μ=0, σ=20, K=10). We design
  and train policies here.
- **Van Roy reproduction instances** (`van_roy_1997` dynamics): `van_roy1997_simple_problem` (51.7),
  `van_roy1997_case_study1` (1302), `van_roy1997_case_study2` (1449). These exist only to reproduce
  the published *absolute* constant base-stock costs and to carry the published A3C savings
  (8.95% / 12.09%).

## Action-space design and the Gijs `{50..100}` ambiguity (important)

A learned policy is only as good as the action space it can express. The base-stock grids carried in
`references.rs` (`GIJS_SETTING_WAREHOUSE_LEVELS = {50,60,…,100}`; retailer `{0,5,…,40}` / `{0,…,50}`)
are transcribed from Gijs's *reduced* learned-policy grid, and **that grid is unfit for this problem
as an installation warehouse base-stock grid.**

The inconsistency: the paper states the learned warehouse base-stock level `y^w_St ∈ {50..100}`, yet
its own constant base-stock benchmark for the *same* setting uses `yw = 330` (setting 1) and
`yw = 460` (setting 2). An installation base-stock of 100 cannot reach the ~300-460 the problem needs.

Empirical confirmation (gijs_2022, setting 1, grid search over constant base-stock):

| warehouse grid | best (yw, yr) | mean cost |
|---|---|---|
| `{50..100} × {0..40}` (Gijs reduced) | (100, 0) | ~3090 |
| `{50..400} × {0..60}` (operating region) | (300, 25) | ~911 |

With the warehouse pinned at 100 and **free expedite** (`cw = 0`, `Pw = 0.8`, `p = 60`), the optimum
collapses to "hold no retailer stock (`yr = 0`), expedite everything for free" — a 3.4× worse,
degenerate regime with no room for a learned policy. Widen the grid to span ~300 and the structure is
sensible again (`(300, 25)`, matching the published `(330, 23)` scale). So `{50..100}` is a
transcription/parameterization artifact (likely an echelon level, an order-quantity grid, or another
parameterization in the paper — the audit flagged the Gijs action space as unresolved); it is **not**
a faithful action space and we do not adopt it as-is.

**Design rule.** Size the warehouse and retailer order-up-to grids (and any continuous action ranges)
to the operating region the cost structure actually drives the system to — for these settings,
warehouse up to ~350-400 and retailer up to ~40-60 — for **both** the constant-base-stock benchmark
and the learned action space. Confirm the grid is not binding at its endpoints before trusting any
policy number.

## Policy class

The learned policy follows the same interface as lost_sales: the env emits the **pure decision
state** (`raw_decision_state`: warehouse on-hand+pipeline, retailer on-hand+pipeline) and the
**policy** normalizes it (`StateNormalizer` divide-by-scale; scale = max order-up-to level) before a
soft tree (oblique splits, linear leaves) produces the order-up-to action. The policy input dimension
is reported by the problem itself (`multi_echelon_policy_feature_dim`), not re-derived in Python.

Action parameterizations available: `direct_base_stock` (state-dependent warehouse + shared retailer
order-up-to), `anchor_adjustment`, `direct_warehouse_order_store_target`.

## Benchmark / reporting protocol

For each instance we report, in one run (`scripts/multi_echelon/train_multi_echelon_policy.py`):

1. **Literature reproduction** (van_roy_1997 instances only): cost at the published levels vs the
   published number (the env-faithfulness check).
2. **In-env best constant base-stock** (grid search over a properly-sized grid) — the benchmark our
   learned policy must beat.
3. **Learned soft-tree** (CMA-ES) and its **relative improvement over the best constant base-stock**.

For the faithful gijs_2022 settings we compare that relative improvement to the **published A3C
savings** (setting 1: 8.95%, setting 2: 12.09%), pulled from the sibling van_roy_1997 instance. If we
beat the published savings, we report it as a result.

## Related literature (the same / closely-related problem)

See `rust/src/problems/multi_echelon/divergent_special_delivery/literature/README.md` for full
citations and carried rows. Papers on this and closely-related divergent two-echelon problems:

- **Van Roy, Bertsekas, Lee & Tsitsiklis (1997)** — the base one-warehouse/K-retailer model with
  special delivery; neuro-dynamic programming. Source of the absolute constant base-stock / NDP rows.
- **Gijsbrechts, Boute, Van Mieghem & Zhang (2022, MSOM)** — A3C DRL on the same two settings; source
  of the relative A3C savings (8.95% / 12.09%).
- **Nahmias & Smith (1994, Mgmt Sci)** — two-echelon retailer system with **partial lost sales**;
  the closest classical analogue (Gijs cites it as closely related).
- **Federgruen & Zipkin (1984, Mgmt Sci)** — approximations for dynamic multi-location (divergent)
  production/inventory; classical divergent structure.
- **de Kok et al. (2018, EJOR 269(3):955–983)** — typology and literature review of stochastic
  multi-echelon inventory models (positions the divergent/lost-sales variants).
- **Kaynov et al. (2024, IJPE 267:109088)** — DRL for the one-warehouse multi-retailer (OWMR)
  divergent system; a modern DRL benchmark on the same topology (see the repo's
  `one_warehouse_multi_retailer` problem).
- **Cheng et al. (2023, WSC)** — reuses the two Van Roy/Gijs 10-retailer settings; reports relative
  improvements (NDP 10%, A3C 9%/12%, RBF-DQN 12%).

When adding a new comparator, prefer those that consider the **same divergent lost-sales/special-
delivery topology**; record absolute rows in `references.rs` and the discussion in `literature/`.

## Budgets and search surface

- Budgets: `screening` / `full` in `scripts/multi_echelon/{autoresearch_multi_echelon,train_multi_echelon_policy}.py`.
- Search surface: the soft-tree policy (`rust/src/core/policies/`), the observation/normalizer and
  action grids (`rust/src/problems/multi_echelon/divergent_special_delivery/rollout.rs`,
  `references.rs`), and the CMA-ES driver (`invman/`).

## Search direction (learnings → priors for the next runs)

Established by the runs so far — treat these as priors; do not re-litigate them:

1. **Action design: `direct_level` ≫ `grid`.** Direct estimation (continuous → non-negative int,
   bounded only by the physical caps) beats the discrete Gijs `{50..100}` grid by a huge margin
   (setting 1: 779.8 vs ~3090). **Default to `direct_level`;** keep `grid` only as an ablation baseline.
2. **Tree depth ≈ 2 is the sweet spot at the current budget.** Depth-2 won; depth-3 underperformed
   *and* costs ~2.5× more to train, so it needs more episodes / a warm start to pay off. Prior:
   depth 2; only push deeper with a larger budget or warm start.
3. **Benchmark over the operating region, never the reduced grid.** Constant base-stock must be
   searched over the physical region (warehouse up to ~500), not `{50..100}`.
4. **Observation = raw decision-state + divide-by-scale** works; `scale = Cr` is a reasonable default.

Next dimensions to add to the sweep (not yet explored), in rough priority order:

- direct **order-quantity** action design (estimate `q^w` bounded by `Cm`) vs direct level;
- **per-retailer** action heads vs the single shared retailer target;
- finer benchmark grid; **depth-3 with warm start / larger budget**; normalizer-scale sweep;
  temperature / split-type.

Objective per faithful setting: maximize the learned policy's improvement over the operating-region
best constant base-stock, and compare to the published A3C savings
(setting 1: 8.95% → **achieved 14.4%**; setting 2: 12.09% → to run).

## Current status

- **Simple problem** (`van_roy1997_simple_problem`): pipeline validated end-to-end. Literature
  reproduction +0.50% at the published `(10,16)`; in-env best constant base-stock 47.69 at `(10,22)`;
  learned soft-tree depth-3 = 47.70 (+0.03% vs best base-stock — matches, as expected for a
  single-retailer near-newsvendor system).
- **gijs_2022 setting 1**: resolved. Autoresearch sweep (`--designs grid,direct_level --depths 2,3`,
  full budget): operating-region best constant base-stock `(300,25)` = 911.4; `grid` policies stay
  stuck at ~3090 (+238%, the `{50..100}` trap); **`direct_level` depth-2 = 779.8, a −14.44%
  improvement over best constant base-stock — exceeding Gijs's published A3C savings of 8.95%.**
  `direct_level` depth-3 (1226, +34%) underperformed in this budget; the design search picked depth-2.
  This is the same relative-improvement metric Gijs reports; absolute costs differ (gijs_2022 mode).
- **gijs_2022 setting 2**: next — run the same sweep (μ=0); expect direct_level to similarly beat the
  benchmark. Optional refinements: finer benchmark grid, add the direct order-quantity and
  per-retailer action designs to the sweep, and give depth-3 more training budget / a warm start.
