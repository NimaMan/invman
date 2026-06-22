# ameliorating_inventory

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "ameliorating_inventory",
  "instance_id": "pahr_grunow2025_spirits_0001",
  "instance_parameters": {
    "source_dataset": "author companion spirits instance 0001"
  },
  "policy": "perfect_information_lp",
  "metric": "max_reward_average_profit_upper_bound",
  "expected_value": 1991.9344293376805,
  "reference": {
    "citation": "Pahr and Grunow (2025), The Value of Blending - Managing Ameliorating Inventory Using Deep Reinforcement Learning, Production and Operations Management",
    "locator": "spirits_0001 companion LP reference row",
    "doi_or_url": "https://doi.org/10.1177/10591478251387795",
    "literature_verified": false,
    "notes": "Author companion-data perfect-information upper bound; not an achieved trainable-env policy value."
  },
  "code_value": 1991.9344293930808,
  "tolerance": {
    "absolute": 1e-06
  },
  "command": "python - <<'PY'\nimport invman_rust as ir\ns = ir.ameliorating_inventory_perfect_info_lp_bound_summary(\"pahr_grunow2025_spirits_0001\")\nprint(s[\"upper_bound_max_reward\"])\nprint(s[\"published_max_reward\"])\nprint(s[\"max_reward_gap_to_published\"])\nassert abs(s[\"upper_bound_max_reward\"] - s[\"published_max_reward\"]) <= 1e-6\nPY"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `reference_companion_bound` |
| Instance | `pahr_grunow2025_spirits_0001` |
| Metric | perfect-information LP upper bound on average profit, `max_reward` |
| Literature / companion value | `1991.9344293376805` |
| Current repo value | `1991.9344293930808` |
| Tolerance | `1e-6` absolute |
| Last validated | `2026-06-22` |

### Source

Pahr and Grunow (2025), "The Value of Blending - Managing Ameliorating Inventory Using Deep Reinforcement Learning", Production and Operations Management, DOI `10.1177/10591478251387795`.

The numeric anchor is the public companion-code/data LP bound carried in `references.rs`, not an achieved trainable-env policy value. The trainable environment is still faithful/unverified against a published achieved-cost or achieved-profit number.

### Validation command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.ameliorating_inventory_perfect_info_lp_bound_summary("pahr_grunow2025_spirits_0001")
print(s["upper_bound_max_reward"])
print(s["published_max_reward"])
print(s["max_reward_gap_to_published"])
assert abs(s["upper_bound_max_reward"] - s["published_max_reward"]) <= 1e-6
PY
```

### Notes

Use this as an upper-bound reproduction check only. A learned policy can be compared to the bound as a gap-to-bound result, but matching or beating the bound is not the right expectation because it is perfect-information.

Rust-first problem home for `ameliorating_inventory`.

Source paper: Pahr & Grunow (2025), "The Value of Blending — Managing Ameliorating Inventory
Using Deep Reinforcement Learning", Production and Operations Management 35(5),
DOI 10.1177/10591478251387795. Companion code:
https://github.com/amelioratinginventory/ameliorating_inventory

## Status

- literature-verified: TRUE
- An EXECUTING in-crate test re-solves the companion perfect-information LP from a checked-in
  dataset and reproduces the published average-profit upper bound (`max_reward`) for two anchors:
  - `spirits_0001` (10 ages, 3 products, target ages [2,4,6], capacity 50, holding 2.5, no
    blending): published `max_reward = 1991.9344293376805`, re-solved gap < 1e-7.
  - `port_wine` (25 ages, 2 products, target ages [9,19], blending enabled): published
    `max_reward = 2444.8010643781136`, re-solved gap < 1e-7.
- This is a reproduction, not a frozen snapshot: `tests/verification.rs` runs the solver and
  asserts the freshly computed value matches the published number within `1e-3`.

## Faithful model (canonical)

The companion environment optimises long-run AVERAGE profit of an age-structured ameliorating
inventory with a price-augmented state and a 3-part action (purchase / production / issuance):

- `average_profit_blending_env.rs` — faithful per-period dynamics: truncated-Normal purchase
  price (mean 200, std 50, truncated +-70), correlated demand/sales price, age-dependent
  stochastic Beta decay (mean = `decay_mean[a]`) plus multiplicative evaporation
  (`(1-evaporation)^(a+1)`), per-age capacity, blending issuance, and the reward
  `revenue - purchase_cost - holding + decay_salvage - outdating`. Step ordering matches the
  companion `step_continuous_issuance_lp`.
- `issuance_blending_lp.rs` — the per-period blending issuance LP (target-age mean constraint,
  blending / no-blending rules, evaporation, production cap), solved with the in-crate `microlp`
  simplex.
- `perfect_information_lp.rs` — the perfect-information (steady-state, expected-value) LP that
  produces the published `max_reward` upper bound. This is the literature-verification anchor.
  Its formulation matches the companion `upper_bound` function line-for-line (objective,
  inventory balance, outdating, target-age, blending). Break points of the piecewise-linear
  revenue envelope are clamped to their valid interval `[tp[i], tp[i+1]]` to remove
  finite-precision overshoot (see the algorithmic header in that file).
- `lp_dataset_loader.rs` — parser for the checked-in companion datasets (config + per-product
  expected-revenue / slope tables + published bound) under `practical/datasets/`.
- `references.rs` — literature instances and published anchors:
  `PRIMARY_REFERENCE_INSTANCE` (spirits_0001), `PORT_WINE_REFERENCE_INSTANCE`, and
  `VERIFICATION_PROBLEM_INSTANCE` (the spirits_0001 upper-bound anchor).

## Reduced model (retained tractable companion, NOT the verification target)

`env.rs`, `issuance.rs`, `rollout.rs`, `heuristics/`, `finite_horizon_dp.rs`, `bindings.rs`,
`demand.rs`, and `literature/` implement an earlier discrete, discounted-cost approximation used
by the soft-tree rollout path. It is kept for the existing Python bindings and its own exact
worked-example verifier (`verification/tests.rs`), but it is no longer the canonical formulation.

## Package layout

- canonical faithful env: `average_profit_blending_env.rs`
- issuance LP: `issuance_blending_lp.rs`
- perfect-information upper-bound LP: `perfect_information_lp.rs`
- dataset loader: `lp_dataset_loader.rs`
- literature references + anchors: `references.rs`
- executing literature verification: `tests/verification.rs`
- checked-in companion datasets: `practical/datasets/`
- reduced-model exact verifier: `verification/tests.rs`, `finite_horizon_dp.rs`
- reduced-model heuristics: `heuristics/`
- reduced-model rollout path: `rollout.rs`
- experiment notes: `experiments/`
