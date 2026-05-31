# Literature

## Classical formulation anchor (what the env implements)

The executable env is a classical **finite-horizon joint pricing-and-inventory** model whose single-
period reduction is the **price-setting newsvendor** (overage `Co = c + h`, underage `Cu = p + s − c`,
critical-fractile order-up-to). Carried in `references.rs` as `PRICE_SETTING_NEWSVENDOR_ANCHOR`:

- Whitin (1955), inventory control and price theory (the original price-setting newsvendor)
- Petruzzi & Dada (1999), pricing and the newsvendor problem
- Federgruen & Heching (1999), combined pricing and inventory control under uncertainty
  (`https://doi.org/10.1287/opre.47.2.183`) — the canonical finite-horizon joint model

This is the one anchor that is **independently verified**: `verification/tests.rs` checks the env's
`T = 1` optimum equals the closed-form critical fractile for every price (an analytical, not self-
consistency, check).

## DRL / data-driven anchors (formulation-class only, no reproduced numbers)

- Zhou et al. (2022), DRL for joint pricing and inventory under reference-price effects
  (`https://doi.org/10.1016/j.eswa.2022.116564`)
- Qin, Simchi-Levi & Wang (2022), data-driven approximation schemes for joint pricing and inventory
  control (`https://doi.org/10.1287/mnsc.2021.4212`)

## Current status: this package is NOT literature-verified

Why (pinned root cause):

- **Zhou et al. (2022)** use an **infinite-horizon MDP with a reference-price state**
  (adaptation-level theory); the repo deliberately omits that state, so it is a different MDP.
- **Qin et al. (2022)** match the repo's model *class* (finite-horizon, profit, price-dependent
  demand) but prove a sample-complexity theorem for a data-driven SAA scheme; the publicly accessible
  article does not expose a reusable per-instance optimal-profit table to anchor to.
- The benchmark-policy names carried in `references.rs` for both papers
  (`ddqn_joint_price_inventory`, `value_iteration_baseline`, `q_learning_baseline`,
  `data_driven_approximation`, `deterministic_baseline`, `random_baseline`) are **labels only — none
  are implemented in this package**, so no published number is reproduced.

So the repo uses Zhou/Qin as formulation-class anchors, the classical newsvendor results as the
analytical verification anchor, and a repo-native reduced exact DP for implementation correctness.

Reference hygiene:

- `references.rs` stores literature anchors and problem-instance definitions only
- repo-native worked-transition expected values live in verification tests, not in literature references
