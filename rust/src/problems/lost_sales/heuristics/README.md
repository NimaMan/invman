# Lost-sales heuristics

Self-contained, reusable implementation of the three vanilla lost-sales ordering
heuristics — **Myopic-1**, **Myopic-2**, and the **Standard Vector Base Stock
(SVBS)** policy — plus a rollout-based average-cost evaluator. This module is the
single source of truth for the heuristic policy logic; the flownet verification
module (`flownet/verification/policy_performance.rs`) re-exports from here rather
than holding its own copy.

## Folder contents

| File | Functionality |
| --- | --- |
| `mod.rs` | Module entry point and full algorithmic description. Re-exports the public API, defines the canonical `VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG` instance and its reference/horizon/seed constants, and holds the literature-number verification tests. |
| `policy_kind.rs` | `LostSalesHeuristicPolicyKind` enum (`Myopic1`, `Myopic2`, `StandardVectorBaseStock`) with `policy_name()` (`"myopic1"`/`"myopic2"`/`"svbs"`) and `all()`. |
| `demand_support.rs` | Demand-law utilities: truncated/normalised per-period Poisson & Geometric PMFs, the MMPP2 **stationary marginal** PMF (`markov_modulated_poisson2_stationary_marginal_support`, routed through `iid_demand_support`), and the multi-period cumulative demand CDF (`cumulative_demand_cdf`) used by SVBS. |
| `evaluator.rs` | `LostSalesHeuristicEvaluator` and `LostSalesHeuristicVerificationConfig`. Computes order quantities for each heuristic with memoised one-period costs, lookahead values (`q_l`), Myopic-2 values, and SVBS base-stock levels. |
| `rollout.rs` | `PolicyPerformanceMeasurement`, `measurement_from_observed_mean_cost`, and the public `evaluate_heuristic_policy(config, policy)` entry point that simulates the lost-sales environment and returns the warm-up-adjusted mean cost. |

## Public API

- `LostSalesHeuristicPolicyKind` — selects which heuristic to run.
- `LostSalesHeuristicVerificationConfig` — per-instance config (lead time, costs,
  demand, horizon, seed, warm-up ratio, order-search bound, discount factor).
- `LostSalesHeuristicEvaluator::new(config)` and its order-quantity methods
  (`myopic_1_order_quantity`, `myopic_2_order_quantity`,
  `standard_vector_base_stock_order_quantity`, `standard_vector_base_stock_levels`).
- `evaluate_heuristic_policy(config, policy) -> PolicyPerformanceMeasurement` —
  the main entry point: roll out one heuristic and report its average cost.
- `VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG` and the matching
  `_REFERENCE` / `_HORIZON` / `_SEED` constants — the canonical instance.

## The three heuristics (summary)

- **Myopic-1** — single lead-time-deep newsvendor lookahead; orders the `z` that
  minimises the expected discounted cost assuming no further ordering.
- **Myopic-2** — Myopic-1 action value plus a discounted one-step continuation
  equal to the expected Myopic-1 quantity chosen next period. Two-period view;
  usually the best of the three.
- **SVBS** — vector base-stock levels, one per pipeline position, set to the
  critical-fractile `(c_p + c_h)/(c_h + c_s)` quantile of the corresponding
  multi-period demand; orders up to the tightest binding level.

See the comment block at the top of `mod.rs` (and `evaluator.rs`) for the full
algorithmic description, including the pipeline-state dynamics and the
`best_quantity` line search.

## Verification

`evaluate_heuristic_policy` on `VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG`
reproduces the trusted literature average costs:

| Policy | Expected | Measured |
| --- | --- | --- |
| myopic2 | 4.82 | 4.8208 |
| myopic1 | 5.06 | 5.0569 |
| svbs | 5.83 | 5.8153 |

with ordering `myopic2 <= myopic1 <= svbs`. The tests
`vanilla_heuristic_mean_costs_match_literature_numbers` and
`vanilla_heuristic_ordering_holds_myopic2_is_best` in `mod.rs` assert these.

## Markov-Modulated Poisson (MMPP2) demand

The heuristics also accept `MarkovModulatedPoisson2` demand. The order-quantity
math requires a closed-form per-period demand law, so MMPP2 is handled via its
**stationary marginal**:

```
P(d) = prob_low * Poisson(d; lambda_low) + prob_high * Poisson(d; lambda_high)
```

where `prob_low`/`prob_high` are the stationary regime occupancies
`prob_high = (1 - p00) / (2 - p00 - p11)`, `prob_low = 1 - prob_high`. Each
Poisson component is truncated with the same tail cutoff as the IID Poisson
support, then mixed and renormalised. SVBS's multi-period lead-time demand is the
self-convolution of this marginal (periods treated as independent — the regime
autocorrelation is intentionally ignored).

Important: the stationary marginal is used **only** to choose order quantities.
The rollout cost is always measured on the true autocorrelated MMPP2 process
(sampled via `build_demand_process`/`sample_demand`), so the reported mean cost
is a valid "heuristic-evaluated-on-true-environment" number. These MMPP2 numbers
are **repo-computed, not from the literature** (which only covers IID demand).
The test `mmpp2_heuristics_run_and_return_finite_positive_costs` in `mod.rs`
runs all three heuristics on the `lit_mmpp2_pos_p4_l4` instance (lambda_low=3,
lambda_high=7, p00=p11=0.9, lead time 4, holding 1, shortage 4) and asserts they
complete with finite, positive, roughly-ordered costs.

## Running the tests

The crate is a pyo3 extension module (`crate-type = ["cdylib"]` with the
`extension-module` feature always on), so the test binary must be told where to
find `libpython` — otherwise linking fails with `undefined symbol: Py...`. Adjust
the libpython path/version for your system:

```bash
RUSTFLAGS="-L /usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" \
  cargo test heuristics

# Or the whole lost-sales suite (heuristics + flownet verification):
RUSTFLAGS="-L /usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" \
  cargo test lost_sales::
```
