# Verification

The verification target for this family is a reduced finite-horizon joint pricing-and-ordering instance with:

- a three-level price ladder
- price-specific demand distributions
- periodic ordering with lost sales
- exact finite-horizon DP as the reference solver

There are two kinds of check. Only the first is an *independent* (closed-form) check; both confirm
implementation correctness, not literature verification.

## 1. Analytical anchor — price-setting newsvendor critical fractile (independent)

`single_period_env_matches_price_setting_newsvendor_critical_fractile` confirms that the env's `T = 1`
reduction matches the textbook price-setting newsvendor closed form: for each price, the brute-force
expected-profit-maximizing order quantity computed *through `step_state`* equals the critical-fractile
order-up-to `smallest y with F(y) ≥ Cu/(Cu+Co)`, with overage `Co = c + h` and underage
`Cu = p + s − c`. This validates the env transition + cost against an independent classical result.

Confirmed numerically against the installed bindings on `VERIFICATION_PROBLEM_INSTANCE`:
prices `(7, 9, 11)` → critical-fractile `y* = (3, 2, 2)`, matched by env brute force.

## 2. Reduced exact DP (self-consistency)

The repo finite-horizon DP (`finite_horizon_dp.rs`) is checked to dominate both heuristics on the
verification instance (`exact_dp_dominates_repo_heuristics`). It was additionally cross-checked
exactly against an independent Python DP through the bindings: optimal discounted cost `−33.1781`,
optimal first action `(2, 1)`.

This verifier is repo-native. It checks implementation correctness, not literature verification —
no published per-instance optimal-profit row is reproduced (see `literature/README.md` for why).

Repo-native worked-transition expected values are kept in the verification tests, not in
`literature/references.rs`.

The executable assertions live in `rust/src/problems/joint_pricing_inventory/verification/tests.rs`.
