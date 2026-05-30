# Divergent Special Delivery

This subfolder implements a Rust version of the original two-echelon warehouse-plus-stores formulation
from Van Roy et al. (1997). The later Gijsbrechts et al. (2022) benchmark is treated as a reduced-action
benchmark family built on the same case-study settings.

## Formulation

The system has one warehouse and `K` identical stores.

- `q_{0,0}` is on-hand warehouse inventory.
- `q_{0,T}` for `T = 1, ..., D_w` is inventory that will arrive at the warehouse in `T` days.
- `q_{i,0}` is on-hand inventory at store `i`.
- `q_{i,T}` for `T = 1, ..., D_s` is inventory that will arrive at store `i` in `T` days.

The pre-decision state is the full buffer vector

- `x_t = (q_{0,0}, ..., q_{0,D_w}, q_{1,0}, ..., q_{1,D_s}, ..., q_{K,0}, ..., q_{K,D_s})`.

At each decision epoch the controller chooses

- a warehouse order `a_0`
- store replenishment orders `a_1, ..., a_K`

subject to production, warehouse-capacity, store-capacity, and current-warehouse-availability
constraints.

The Van Roy event order implemented here is:

1. the warehouse order enters the inbound warehouse pipeline
2. store replenishment orders are transferred out of current warehouse stock into the outbound store
   pipelines
3. store demand is realized; unmet customers request same-day warehouse service independently with
   probability `P_w`
4. transportation buffers advance one stage

Demand is sampled independently across stores as:

- `z_i ~ N(mu, sigma)`
- `d_i = max(0, ceil(z_i - 1/2))`

The period cost has three components only:

- warehouse holding cost on warehouse on-hand inventory
- store holding cost on store on-hand inventory
- shortage cost plus special-delivery cost

Regular store replenishment does not incur a separate transportation-charge term in this model.

## Heuristics And Published Policy Rows

Van Roy uses a constant `s`-type benchmark policy with:

- one warehouse order-up-to level
- one shared store order-up-to level

Store orders are computed from store inventory positions. If current warehouse stock cannot satisfy
all desired store orders, the intended Van Roy allocation rule maximizes the minimum resulting store
inventory position across stores. The warehouse order is then chosen after accounting for those store
shipments.

The published benchmark rows carried in `references.rs` are:

- `van_roy1997_simple_problem`
  - constant base-stock `(10, 16) -> 51.7`
  - best reported NDP row `52.6`
- `gijsbrechts2022_setting1`
  - this is Van Roy case study 1 reused by Gijs
  - constant base-stock `(330, 23) -> 1302`
  - best reported NDP row `1179`
- `gijsbrechts2022_setting2`
  - this is Van Roy case study 2 reused by Gijs
  - constant base-stock `(460, 22) -> 1449`
  - best reported NDP row `1318`

For the two complex case studies, Van Roy also reports the reduced NDP action space used in the
paper:

- warehouse order grid `{50, 60, ..., 100}`
- store target grid `{0, 5, ..., 40}` in case study 1
- store target grid `{0, 5, ..., 50}` in case study 2

The current Rust rollout supports both families:

- constant base-stock over fixed heuristic levels
- soft-tree policies over the reduced Van Roy decision grid

The important benchmark distinction is:

- the published Van Roy constant base-stock benchmark row uses the published heuristic levels
  `(330, 23)` and `(460, 22)` for the two carried case studies
- the reduced grid `{50, 60, ..., 100}` plus `{0, 5, ...}` belongs to the learned-policy action
  space, not to the published constant base-stock benchmark row

## Literature

Primary references:

- Van Roy et al. (1997), full report: <https://www.stanford.edu/~bvr/pubs/retail.pdf>
- Van Roy et al. (1997), CDC version: <https://www.mit.edu/~jnt/Papers/C-97-bvr-retail-CDC.pdf>
- Gijsbrechts et al. (2022): <https://doi.org/10.1287/msom.2021.1064>

How this package treats them:

- Van Roy is the literature reference for the executable formulation and the published heuristic/NDP
  rows we want to reproduce.
- Gijs supplies later benchmark settings and published relative improvements over constant
  base-stock on the same family.
- `literature_verified` applies to repo heuristic or exact implementations only, not to published
  A3C/NDP rows carried from the papers.

## Verification

This package carries two different validation layers.

1. Literature validation for the original Van Roy family:
   - attempt to reproduce the published constant base-stock rows on the simple problem and the two
     complex case studies
   - compare trained policies against the published NDP rows
2. Repo-native exact verification:
   - `VERIFICATION_PROBLEM_INSTANCE`
   - a reduced finite-horizon verifier used to compare repo heuristics and learned policies to the
     true optimum on a tractable easy instance

Current status:

- the literature rows are carried in `references.rs`
- the repo heuristic is **not** literature-verified yet
- the current reproduction results are recorded in `outputs/multi_echelon/van_roy_validation_2026-04-10.json`
- the protocol audit is recorded in `outputs/multi_echelon/van_roy_protocol_audit_2026-04-12.json`
- the exact verifier is `literature_verified = false`; it exists to validate the Rust implementation,
  not to support literature claims

## Policy Interface

The environment interface remains raw-state first:

- `env.rs` exposes raw inventory and pipeline state
- feature construction happens in `rollout.rs`
- normalization belongs in rollout or policy code, not in the environment

For learned policies, the Rust rollout currently supports these observation modes:

- `raw_decision_state` (default for training) — the **pure** decision-state observation
  (warehouse on-hand+pipeline, retailer on-hand+pipeline), UNNORMALIZED. This mirrors the
  lost-sales policy interface: the env emits the raw state and the **policy** normalizes it.
- `full_decision_state` — the same layout, but pre-normalized by the inventory caps inside
  the feature builder (normalization baked into feature construction).
- `symmetric_summary`
- `compact_summary`

`compact_summary` is a 22-feature engineered summary; it is not a literature-verified
formulation boundary.

Observation normalization is a **policy-owned** step (`StateNormalizer`, mirroring
`lost_sales::env::StateNormalizer`), applied to the observation in the rollout before the
soft tree acts:

- `identity` — leave the observation as-is (used by the pre-normalized feature modes)
- `divide_by_scale` — divide the raw observation by a single positive `state_scale`

The Python policy builder sets the multi-echelon policy to `raw_decision_state` +
`divide_by_scale`, with `state_scale` = the largest base-stock / order-up-to level across
both echelons (the action magnitude bounding the inventory positions the policy steers to,
the multi-echelon analogue of lost_sales' `state_scale = max_order_size`).

For action parameterization, the repo currently supports:

- `direct_base_stock`
  - Gijs-aligned reduced action design: state-dependent warehouse and shared retailer order-up-to
    levels
- `anchor_adjustment`
  - repo-local variant that adjusts around one fixed anchor pair
- `direct_warehouse_order_store_target`
  - Van Roy NDP-style control parameterization: direct warehouse order plus shared retailer target

## Gijs Subfamily

Gijs reuses the two Van Roy complex case studies and reports relative improvements over the constant
base-stock benchmark:

- setting 1 A3C savings: `8.95% +/- 0.13%`
- setting 2 A3C savings: `12.09% +/- 0.39%`

Those rows are comparison targets for later experiments. They are not the primary literature
verification reference for the heuristic implementation because the stronger absolute benchmark rows are
already available from Van Roy.
They are, however, the Gijs-specific literature-number target for audits:
the metric is `published_relative_a3c_savings_vs_constant_base_stock_pct`. This is not an
implementation verification unless a repo policy reproduces the published relative row or the repo
simulator reproduces the underlying absolute Van Roy heuristic rows.

Use `verification::van_roy_reproduction_summary` or the Python binding
`invman_rust.multi_echelon_van_roy_reproduction_summary(...)` for the strict absolute check against
the published Van Roy constant base-stock rows. Use
`verification::gijs_relative_verification_summary` or
`invman_rust.multi_echelon_gijs_relative_verification_summary(...)` only to inspect the carried
paper-relative A3C rows. The existing soft-tree paper benchmark is exploratory and should not be
treated as an executable reproduction of the published A3C learner.

## Implementation Review

The current Rust model matches the Van Roy appendix reasonably closely for the carried benchmark
instances:

- state layout follows the warehouse-plus-store buffer vector in Appendix A
- action constraints follow the warehouse capacity, store capacity, and current warehouse inventory
  constraints
- demand is sampled as rounded-and-clipped normal demand
- special deliveries are modeled as independent Bernoulli acceptances for unmet units
- cost uses storage, shortage, and special-delivery charges only

Known review findings:

- `initialize_random_state` currently initializes the heuristic simulator from the zero state; this
  is a repo evaluation choice, not a literature-verified protocol
- Van Roy states only that heuristic rows were computed from a "lengthy simulation"; the paper does
  not give a single explicit heuristic warm-up ratio or initial-state convention
- Van Roy's NDP figures use rolling finite windows during one long training run:
  `10,000` steps in the simple problem and `5,000` steps in the two complex case studies
- the simple-problem mismatch is partly parameterization-sensitive: evaluating the published
  `(10, 16)` policy under the induced rounded-demand moments `(6.2, 6.2)` nearly matches the
  published `51.7`, while the latent normal parameters `(5, 8)` do not
- the current package has not yet matched the published Van Roy heuristic rows under one stable
  evaluation protocol, so `literature_verified` remains `false`
- direct zero-delay edge cases are not part of the carried Van Roy benchmarks and should not be
  treated as validated semantics
