# program_ameliorating_inventory — autoresearch for the faithful average-profit env

The objective of this program is one honest learned-policy result on the **faithful
long-run average-profit** ameliorating-inventory env
(`src/problems/ameliorating_inventory/average_profit_blending_env.rs`, the
Pahr & Grunow 2025 model), scored by the new Rust population-rollout binding under paired
CRN and reported as a **GAP-TO-BOUND %** against the perfect-information LP upper bound.

## Trusted benchmark (fixed)

- Instances: **spirits_0001** (10 ages, 3 products, target ages [2,4,6], no blending,
  bound 1991.9344293376805) and **port_wine** (25 ages, 2 products, target ages [9,19],
  blending, bound 2444.8010643781136). Config fields come from the checked-in LP datasets
  (`practical/datasets/<instance>_perfect_information_lp.txt`); the env-only fields
  (demand/sales-price/correlation/decay CoV, NOT used by the LP bound) come from the
  companion `config.json` verbatim, recorded in `literature/references.rs`.
- Baseline = the **perfect-information LP UPPER BOUND on average profit** (`max_reward`),
  reproduced by `perfect_information_lp.rs` to <1e-3 (`tests/verification.rs`). The paper
  reports DRL within ~3.5% of this bound on the generic instance set.

## The binding (call-bridge added)

- `ameliorating_inventory_average_profit_soft_tree_population_rollout`
  (`src/problems/ameliorating_inventory/bindings.rs`) — targets the FAITHFUL
  `average_profit_blending_env.rs` (NOT the reduced discounted-cost `env.rs`). Decodes a
  soft-tree policy into the per-period purchase volume and rolls out `step_state` in Rust
  (`average_profit_rollout.rs`), returning per-individual long-run AVERAGE PROFIT under
  paired CRN. Registered alongside the existing reduced-env bindings.

## Action geometry (the policy)

In the faithful env `step_state` the only free control is the scalar **purchase** volume
`aP ∈ [0, maxInventory]`; the issuance plan is solved by the env's per-period blending LP
and production is derived from it (the "3-part action" is structural). The policy carries a
single purchase head over the price-augmented state `[price, inventory[0..A]]` (normalised
by maxInventory), via the new continuous soft-tree head
`action_vector_continuous_from_flat_params`. A **linear leaf** lets it express a
price-reactive order-up-to purchase. Warm-start = order-up-to:
`purchase = softplus(S_target − sum(inventory))` (bias = S, inventory weights = −maxInventory),
so generation 0 reproduces a simple order-up-to heuristic; the optimizer refines a
price-reactive purchase.

## What we know (autoresearch outcome)

- The price-reactive learned purchase **robustly beats** the best tuned order-up-to
  purchase on BOTH instances at full budget (spirits_0001: 115.07 vs 20.91; port_wine:
  505.78 vs 133.78), with the learned − gate gap ~140x / ~610x the paired-CRN SEM — the
  soft tree exploits the truncated-Normal purchase price (buy more when price is low). This
  is the keep/discard gate (the in-repo heuristic), and it flips a clear win.
- **Gap-to-bound is large and reported honestly.** A single-purchase policy on the
  stochastic env sits 94.2% (spirits_0001) / 79.3% (port_wine) below the perfect-information
  LP bound: the bound assumes full LP issuance from inventory held at every age up to
  capacity, while the faithful env charges the full purchase cost (price ~200/unit) every
  period and issues only from inventory aged into the target ages. port_wine's tighter gap
  is the value of blending (issuance across target ages [9,19]). The reported gap is
  therefore NOT comparable to the paper's 3.5% DRL gap (which uses the full 3-part action
  incl. production targets); the deliverable is the binding plus the honest learned-vs-bound
  and learned-vs-heuristic numbers, not a claim of matching the paper.

## Result — FULL BUDGET, both instances (committed run, `outputs/` is gitignored)

Full budget = popsize 16, 60 generations, 4,000 train periods, 12,000 eval periods, 24
held-out paired-CRN eval seeds; depth-1 oblique soft tree, linear leaf, warm-started at
order-up-to. `gap vs gate (abs)` = learned − best-tuned-order-up-to profit; `(% of gate)` =
the same as a fraction of the heuristic baseline.

| instance | budget | learned ± SEM | LP bound | order-up-to gate | gap vs gate (abs) | gap vs gate (% of gate) | gap to bound | verdict |
|---|---|---|---|---|---|---|---|---|
| spirits_0001 | full | **115.07 ± 0.44** | 1991.93 | 20.91 | **+94.16** | +450% | 94.22% | **beats_order_up_to** |
| port_wine | full | **505.78 ± 0.59** | 2444.80 | 133.78 | **+372.00** | +278% | 79.31% | **beats_order_up_to** |

Paired SEM of the learned−gate difference: 0.66 (spirits_0001), 0.61 (port_wine) — both
beats are ~two orders of magnitude beyond the paired SEM. (Earlier screening-budget runs:
spirits_0001 learned 100.54 vs gate 20.79; the prior table's "+79.8%" was the **absolute**
gap mislabeled as a percent and is corrected here.)

The price-reactive single-purchase soft tree robustly **beats the literature order-up-to
heuristic** (the in-repo gate) on both instances. The large gap to the perfect-information
LP bound is expected and structural — a single purchase action vs the bound's full 3-part LP
issuance — and is NOT comparable to the paper's ~3.5% DRL gap.

**SCOPE / RECOMMENDED FOLLOW-UP (recorded, not implemented here).** Closing the gap-to-bound
requires widening the action head to the full 3-part action (purchase + production targets +
per-age issuance). That needs NEW Rust rollout code and a rebuild of `invman_rust`, which is
OUT OF SCOPE for this run (a rebuild would corrupt the parallel training agents sharing the
install). It is the recommended deeper follow-up to chase the paper's ~3.5% DRL gap.

Runner: `scripts/ameliorating_inventory/autoresearch_ameliorating_inventory_average_profit.py`.
Results write-up: `scripts/ameliorating_inventory/RESULTS_FULL_BUDGET.md`.
Paper draft: `scripts/ameliorating_inventory/PAPER_SECTION_DRAFT.md`.
Ledger / JSON: `outputs/autoresearch/ameliorating_inventory_average_profit_autoresearch/`.
