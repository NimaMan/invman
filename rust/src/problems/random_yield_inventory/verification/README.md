# Verification

`random_yield_inventory` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `literature/references.rs`
- policy-state layout checks (`build_raw_state`)
- worked-transition accounting checks (`step_state`)
- exact reduced finite-horizon DP comparison against the carried heuristics (LIR, WNH)

This verifier is **repo-native / implementation-correctness only**. It checks that the code is correct
on the reduced exact instance; it is **not** a literature-verification claim (no public number exists
to assert against — see `literature/README.md`).

## Independent cross-check performed during the 2026-05 review

- The exact DP (`finite_horizon_dp.rs`) was re-derived from scratch in an independent Python DP of the
  same MDP. It reproduces `optimal_discounted_cost = 40.0598976099` and `optimal_first_action = 4` on
  `VERIFICATION_PROBLEM_INSTANCE` exactly. Raising the DP action cap 8 -> 20 moves the optimum only at
  the 5th significant figure (`40.0598742583`), confirming the carried cap is effectively non-binding
  for the optimal policy and the reported optimum is the true unconstrained optimum to ~5 sig figs.
- The uncapped env rollout (`heuristics/mod.rs`) was Monte-Carlo cross-checked against the exact DP
  heuristic values: with the DP's `max_order_quantity` clamp applied, the env WNH cost converges to
  `60.38 ± 0.08`, matching the DP's `60.3936`. Without the clamp it reads `60.75` — because the WNH
  sometimes wants to order above the cap and the env (correctly) has no physical cap. The clamp is a
  DP-tractability truncation, not a model feature; the `optimal <= heuristic` assertions in
  `tests.rs` hold regardless.

The reduced exact verifier stores only the problem instance in the literature catalog. Optimal,
heuristic, and worked-transition accounting values are generated/asserted in verification code, not
stored as literature reference numbers.
