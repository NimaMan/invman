# Literature

## Classical formulation anchor (what the env implements)

The executable env is a classical **finite-horizon joint pricing-and-inventory** model whose single-
period reduction is the **price-setting newsvendor** (overage `Co = c + h`, underage `Cu = p + s − c`,
critical-fractile order-up-to). Carried in `references.rs` as `PRICE_SETTING_NEWSVENDOR_ANCHOR`:

- Whitin (1955), "Inventory Control and Price Theory", Management Science 2(1):61-68
  (`https://doi.org/10.1287/mnsc.2.1.61`) — the original price-setting newsvendor
- Petruzzi & Dada (1999), "Pricing and the Newsvendor Problem: A Review with Extensions",
  Operations Research 47(2):183-194 (`https://doi.org/10.1287/opre.47.2.183`)
- Federgruen & Heching (1999), "Combined Pricing and Inventory Control Under Uncertainty",
  Operations Research 47(3):454-475 (`https://doi.org/10.1287/opre.47.3.454`) — the canonical
  finite-horizon joint model

Note: the single DOI stored on the `PRICE_SETTING_NEWSVENDOR_ANCHOR.url` field
(`https://doi.org/10.1287/opre.47.2.183`) resolves to the Petruzzi & Dada (1999) review, one of the
three anchor papers; the Federgruen & Heching (1999) DOI is `10.1287/opre.47.3.454` and the
Whitin (1955) DOI is `10.1287/mnsc.2.1.61`.

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

- **Zhou et al. (2022)** formulate the problem as an **MDP with a reference-price state**
  (adaptation-level theory) solved with a Double-DQN variant (TN-DDQN); the repo deliberately omits
  that reference-price state, so it is a different MDP.
- **Qin et al. (2022)** match the repo's model *class* (finite-horizon, profit, price-dependent
  demand) but prove a sample-complexity theorem for a data-driven SAA scheme; the publicly accessible
  article does not expose a reusable per-instance optimal-profit table to anchor to.
- The benchmark-policy names carried in `references.rs` for both papers
  (`ddqn_joint_price_inventory`, `value_iteration_baseline`, `q_learning_baseline`,
  `data_driven_approximation`, `deterministic_baseline`, `random_baseline`) are **labels only — none
  are implemented in this package**, so no published number is reproduced.

So the repo uses Zhou/Qin as formulation-class anchors, the classical newsvendor results as the
analytical verification anchor, and a repo-native reduced exact DP for implementation correctness.

### Per-anchor verifiability ledger (be precise)

| Anchor | Real publication? | What is verified here |
| --- | --- | --- |
| Whitin (1955), Petruzzi & Dada (1999), Federgruen & Heching (1999) | Yes (DOIs above, Crossref-confirmed) | **Analytical (independent)**: env `T=1` optimum equals the closed-form critical fractile `smallest y with F(y) ≥ Cu/(Cu+Co)` for every price on `VERIFICATION_PROBLEM_INSTANCE`. Confirmed values: prices (7, 9, 11) → y* = (3, 2, 2). This is a closed-form check of the env cost structure, NOT a reproduced published per-instance number. |
| Qin, Simchi-Levi & Wang (2022) | Yes (DOI above, Crossref-confirmed) | **Formulation-class only**: same model class (finite-horizon, profit, price-dependent demand); no published per-instance optimal-profit number is stored or reproduced. |
| Zhou et al. (2022) | Yes (DOI above, Crossref-confirmed) | **Formulation-class only, different MDP** (adds reference-price state); no published number stored or reproduced. |
| repo-native reduced exact DP | n/a (no paper) | **Self-consistency**: `finite_horizon_dp.rs` dominates both heuristics; cross-checked against an independent Python DP (optimal discounted cost −33.1781, first action (2, 1)). No public anchor. |

Bottom line: `literature_verified = false` is the honest status. There is a genuine **independent
analytical** anchor (classical newsvendor closed form) plus **self-consistency** against a reduced
exact DP, but **no published benchmark number is reproduced** (the case is "faithful model with no
reusable published per-instance anchor", not a model bug).

Reference hygiene:

- `references.rs` stores literature anchors and problem-instance definitions only
- repo-native worked-transition expected values live in verification tests, not in literature references
