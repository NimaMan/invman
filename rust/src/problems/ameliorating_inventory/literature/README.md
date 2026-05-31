# Literature

## Source paper

- Alexander Pahr and Martin Grunow (2025), *The Value of Blending — Managing Ameliorating
  Inventory Using Deep Reinforcement Learning*, Production and Operations Management.
  DOI: `10.1177/10591478251387795`.
- Companion code: `https://github.com/amelioratinginventory/ameliorating_inventory`
  (gymnasium env `AmelioratingInventoryPOM.py`, RLlib APO actor-critic, LP-based benchmarks,
  per-instance config and perfect-information upper bounds).

## Verification status: NOT literature-verified

The current Rust package is a tractable, internally self-consistent **reduction** of the paper's
model, not a faithful executable port. It is verified only against the repo's own exact
finite-horizon DP (`verification/tests.rs`). No published number anchors any executable assertion.

## Precise fidelity gap (Rust vs. Pahr and Grunow 2025)

| Dimension | Paper / companion repo | Current Rust env | Evidence |
| --- | --- | --- | --- |
| Objective | long-run **average profit**; reported as gap to a perfect-information LP upper bound | finite-horizon **discounted cost** | `rollout.rs:127-131` |
| Action | three subspaces: purchasing `aP`, production `aY_w` per product, issuance `aX_i` per age | **1-D purchase only**; production + issuance collapsed into an exact average-age search | `rollout.rs:63-67`, `issuance.rs` |
| Purchase price | **stochastic** (truncated, mean 200, std 70 trunc on σ=50), carried in state | fixed `purchase_cost_per_unit` | `env.rs:159`, `spirits_0001/config.json` |
| Sales price | **stochastic** (means [250,350,500], CoV 0.1), Gaussian-copula correlated with demand | fixed `product_prices` | `env.rs:189-194`, `config.json` |
| Decay | age-dependent **stochastic beta** proportions plus 0.03 evaporation | fixed deterministic `age_retention` (rounded survivors) | `env.rs:208-214`, `config.json` |
| Processing capacity | capacity `k` (largest profit driver in the paper, Fig. 6) | none | absent in `env.rs` |
| Scale | generic 10 ages / 3 products (target ages [2,4,6]); port wine 25 ages | primary 5 ages / 2 products (target ages [1,3]) | `references.rs` |
| Demand | Gaussian-copula correlated, continuous | independent Poisson (or deterministic) | `demand.rs` |

These are structural model differences, not localized bugs. The amelioration mechanic itself
(units age up one class per period, oldest class absorbing) and the average-age blending issuance
(a blend's mean age must be at least the product target age; young + old stock may be combined to
reach an older target) are present and behave correctly in the reduced model.

## Recorded published anchors (provenance only, non-anchoring)

Carried in `literature/references.rs`. These are real published numbers but the reduced env cannot
reproduce them, so `anchors_repo_assertion = false` on every one:

| anchor | instance | ages / products | reported value |
| --- | --- | --- | --- |
| `PAHR_GRUNOW_2025_SPIRITS_0001_UPPER_BOUND` | `spirits_0001` | 10 / 3 | upper-bound avg profit ≈ 1991.93 |
| `PAHR_GRUNOW_2025_PORT_WINE_UPPER_BOUND` | `port_wine` | 25 / 3 | upper-bound avg profit ≈ 2444.80 |

| `PAHR_GRUNOW_2025_PERFORMANCE` field | reported figure |
| --- | --- |
| DRL reduces RLP gap to upper bound by | 16.9% |
| DRL vs. industry heuristic (NVP+VOL) | 27.7% |
| value of average-age blending vs. none | 18.1% |
| value of minimum-age blending vs. none | 8.6% |
| generic learned-policy gap to upper bound | 3.5% |
| port-wine learned-policy gap to upper bound | 2.8% |

## Source-of-truth constants in `literature/references.rs`

- `PAHR_GRUNOW_2025_TITLE` — exact paper title
- `PAHR_GRUNOW_2025_REFERENCE`, `PAHR_GRUNOW_2025_REPOSITORY_REFERENCE` — provenance + gap notes
- `PRIMARY_REFERENCE_INSTANCE` — repo-native benchmark-shaped instance (not published)
- `VERIFICATION_PROBLEM_INSTANCE` — repo-native exact-DP verifier instance (internal check)
- `PAHR_GRUNOW_2025_SPIRITS_0001_UPPER_BOUND`, `PAHR_GRUNOW_2025_PORT_WINE_UPPER_BOUND`
- `PAHR_GRUNOW_2025_PERFORMANCE`

## What would make this literature-verified (deferred — large, out of scope here)

A faithful port requires (in this directory): a price-augmented state, a 3-part action
(purchase / produce / issue), stochastic purchase and sales prices with copula-correlated demand,
age-dependent stochastic beta decay plus evaporation, a processing-capacity constraint, and an
average-profit objective; then reproduce a published upper-bound gap (e.g. the spirits/port-wine
LP bounds above). This is a new env, not an edit to the present one.
