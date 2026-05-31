# joint_pricing_inventory

Rust-first problem home for `joint_pricing_inventory`.

## Formulation

Repo interpretation:

- one item
- one periodic order quantity decision
- one discrete selling-price decision
- stochastic price-sensitive lost-sales demand
- finite planning horizon with terminal salvage value

At period `t`, the state is `(t, I_t)` where `I_t` is on-hand inventory. The action is a pair
`(q_t, p_t)`:

- `q_t` is the order quantity, bounded by `max_order_quantity`
- `p_t` is a discrete index into the available price ladder

Demand in each period is stochastic and price-dependent. Sales are capped by on-hand inventory, so
unmet demand is lost sales. The period objective combines:

- procurement cost
- holding cost on ending inventory
- stockout penalty on lost sales
- terminal salvage value at the horizon

Code lives under `rust/src/problems/joint_pricing_inventory/`.

## Layout

Literature and verification assets live in:

- `literature/references.rs`
- `verification/tests.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

Core executable code remains at the package root:

- `env.rs`
- `demand.rs`
- `heuristics/`
- `finite_horizon_dp.rs`
- `rollout.rs`
- `bindings.rs`

## Verification Status

Current status: `joint_pricing_inventory` is **not literature-verified** (no published per-instance
optimal-profit row is reproduced). It **is** validated against an independent analytical benchmark and
against an independent DP, so the env transition + cost are implementation-correct.

### What the env actually implements (formulation anchor)

The env in `env.rs` is a faithful, classical **finite-horizon joint pricing-and-inventory** model with
zero lead time, price-dependent stochastic demand, lost sales, holding cost on ending inventory, and a
profit objective. Its single-period (`T = 1`) reduction is exactly the textbook **price-setting
newsvendor**:

- per-period profit `= p·min(q, D) − c·q − h·(q − D)⁺ − s·(D − q)⁺`
- overage cost `Co = c + h`, underage cost `Cu = p + s − c`
- optimal order-up-to `= smallest y with F(y) ≥ Cu / (Cu + Co)` (critical fractile)

Classical sources for this formulation: Whitin (1955, Management Science 2(1):61-68,
doi:10.1287/mnsc.2.1.61), Petruzzi & Dada (1999, Operations Research 47(2):183-194,
doi:10.1287/opre.47.2.183), Federgruen & Heching (1999, Operations Research 47(3):454-475,
doi:10.1287/opre.47.3.454). The multi-period finite-horizon version is the classic joint
pricing-inventory control problem, also the model class studied (in a data-driven setting) by Qin,
Simchi-Levi & Wang (2022, Management Science 68(9):6591-6609, doi:10.1287/mnsc.2021.4212). These are
carried in `literature/references.rs` as `PRICE_SETTING_NEWSVENDOR_ANCHOR` and `QIN_2022_REFERENCE`.
(Note: `PRICE_SETTING_NEWSVENDOR_ANCHOR.url` stores the Petruzzi & Dada DOI; Federgruen & Heching is
doi:10.1287/opre.47.3.454.)

### Why it is still `literature_verified = false`

Pinned root cause (this is a no-published-anchor case, not a model bug):

1. Zhou et al. (2022) use an **infinite-horizon MDP with a reference-price state** (adaptation-level
   theory). The repo deliberately omits the reference-price state, so it is a different MDP.
2. Qin, Simchi-Levi & Wang (2022) match the repo's *model class* (finite-horizon, profit, price-
   dependent demand), but their result is a **sample-complexity theorem for a data-driven SAA scheme**;
   the article does not expose a clean reproducible per-instance optimal-profit table to anchor to.
3. The benchmark-policy names carried in `references.rs` for both papers
   (`ddqn_joint_price_inventory`, `value_iteration_baseline`, `q_learning_baseline`,
   `data_driven_approximation`, `deterministic_baseline`, `random_baseline`) are **labels only — none
   are implemented in this package**. They are not reproduced numbers.

So verification rests on two independent, correct anchors instead of a published number:

- an **analytical** anchor: `verification/tests.rs ::
  single_period_env_matches_price_setting_newsvendor_critical_fractile` confirms the env's `T = 1`
  optimum equals the closed-form critical fractile for every price on the verification instance.
  (Confirmed numerically against the installed bindings: y* = 3, 2, 2 for prices 7, 9, 11.)
- a **reduced exact DP** anchor: the repo finite-horizon DP (`finite_horizon_dp.rs`) is checked to
  dominate both heuristics and was cross-checked exactly against an independent Python DP
  (optimal cost −33.1781, first action `(2, 1)`).

## Benchmark Results

Run with `scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py`
(no rebuild, no retrain; uses installed bindings + stored trained params). Profit = −cost.

### Exact-DP-anchored (verifier instance: 5 periods, discrete price-dependent demand)

| Policy | First action (q, price idx) | Discounted cost | Profit | Profit optimality gap |
| --- | --- | ---: | ---: | ---: |
| exact DP optimal | (2, 1) | −33.178 | 33.178 | 0.00% |
| `static_price_base_stock` | (2, 1) | −32.508 | 32.508 | 2.02% |
| `inventory_sensitive_base_stock` | (2, 2) | −27.594 | 27.594 | 16.83% |

### Learned-vs-heuristic (primary instance: 18 periods, Poisson demand; no exact optimum exists)

4096 fresh held-out seeds (base 777000); trained depth-2 oblique/linear soft-tree from
`outputs/joint_pricing_inventory/tree_primary_d2_linear_b8_s123_e120_eval2048.json`.

| Policy | Mean discounted cost | Profit | Std |
| --- | ---: | ---: | ---: |
| soft tree (depth 2) | −216.060 | 216.060 | 32.10 |
| `static_price_base_stock` | −172.635 | 172.635 | 23.84 |
| `inventory_sensitive_base_stock` | −171.513 | 171.513 | 36.62 |

Learned-policy profit improvement over the best heuristic: **+25.15%**.

## Remaining steps

- A TRUE learned-policy optimality gap on the verifier instance (soft tree trained on the verifier
  instance, compared to its exact DP) is not produced here because the Python `SoftTreePolicy` class
  referenced by `scripts/joint_pricing_inventory/common.py` is missing (it was moved into Rust), and a
  fresh CMA-ES pass would contend with parallel builds. The new benchmark script avoids retraining.
- Optional literature upgrade: if a citeable paper with a reproducible finite-horizon
  joint-pricing-inventory optimal-profit instance is located (e.g. a Federgruen–Heching worked
  example), carry that row in `references.rs` and reproduce it to flip `literature_verified = true`.

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps derived demand and price features in `rollout.rs`
- environment code must not hide learned-policy preprocessing
