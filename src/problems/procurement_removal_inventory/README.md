# procurement_removal_inventory

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "procurement_removal_inventory",
  "instance_id": "reduced_exact_verification_instance",
  "instance_parameters": {
    "scope": "finite-horizon discounted MDP"
  },
  "policy": "exact_dynamic_program",
  "metric": "discounted_optimal_cost",
  "expected_value": null,
  "reference": {
    "citation": "No public literature number currently carried for this exact reduced instance",
    "locator": null,
    "doi_or_url": null,
    "literature_verified": false,
    "notes": "Repo-native exact-DP self-consistency anchor; cited papers are formulation context, not this target value."
  },
  "code_value": 31.78026111369698,
  "tolerance": {
    "absolute": 1e-09
  },
  "command": "python - <<'PY'\nimport invman_rust as ir\ns = ir.procurement_removal_inventory_exact_dp_summary()\nprint(s[\"optimal_discounted_cost\"])\nprint(s[\"optimal_first_action\"])\nassert abs(s[\"optimal_discounted_cost\"] - 31.78026111369698) <= 1e-9\nassert list(s[\"optimal_first_action\"]) == [0, 0]\nPY"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `no_public_literature_number_repo_exact_anchor` |
| Instance | reduced exact verification instance |
| Metric | finite-horizon discounted optimal cost |
| Literature value | none currently available |
| Current repo value | `31.78026111369698` |
| Tolerance | `1e-9` against the repo exact DP anchor |
| Last validated | `2026-06-22` |

### Source

Maggiar and Sadighian (2017), "Joint Inventory and Revenue Management with Removal Decisions", SSRN/Amazon Science working paper, is a structural source for the problem class. The repo does not currently carry a public control-only per-instance cost row from that paper.

### Validation command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.procurement_removal_inventory_exact_dp_summary()
print(s["optimal_discounted_cost"])
print(s["optimal_first_action"])
assert abs(s["optimal_discounted_cost"] - 31.78026111369698) <= 1e-9
assert list(s["optimal_first_action"]) == [0, 0]
PY
```

### Notes

This is a strong repo-native regression target, not a literature verification. Future upgrade path: identify a public procurement/removal instance with a printed or companion-code optimal value and add it here.

Rust-first problem home for `procurement_removal_inventory`: a single-item finite-horizon system in
which the controller jointly decides, each period, how much to **purchase** and how much to
**remove** (return to the vendor or liquidate), under a returnable-quota state.

## Formulation

State (pre-decision, in `env.rs`):

- `inventory_level` — units on hand
- `returnable_inventory` — of those, how many are still returnable to a vendor (always
  `returnable_inventory <= inventory_level`)
- `period`

Each period the controller chooses a `(purchase_quantity, removal_quantity)` pair. The event order
in `env.rs::step_state` is:

1. purchase arrives immediately: `inventory += q`, and `min(q, returnable_purchase_cap)` of the
   purchased units enter the returnable pool (the **fixed returnability** contract of Maggiar &
   Sadighian 2017, Section 3.2: a per-period cap on returnable purchases)
2. removal: `removal_quantity` units leave. Returnable units are returned first, the remainder is
   liquidated (this realizes Corollary 1 of the paper: it is never optimal to liquidate a unit that
   could be returned instead)
3. demand realizes; **lost sales** — `sales = min(demand, on_hand)`, unmet demand is lost and
   charged `shortage_cost_per_unit`
4. holding cost is charged on the **ending** on-hand inventory

Per-period cost (`reward = -period_cost`):

```
period_cost = purchase_cost + holding_cost + shortage_cost
              - (return_value * returned_units + liquidation_value * liquidated_units)
```

At the horizon a terminal salvage credit `s*min(x,y) + l*max(x-y,0)` is applied — exactly the
example terminal value `VT(x,y) = s min(x,y) + l max(x-y,0)` of Maggiar & Sadighian (2017),
Assumption 4.

Cost ordering enforced in `env.rs::validate_costs` matches the paper's Assumption 2:
`purchase_cost > return_value` (2.ii) and `return_value >= liquidation_value` (2.iii).

## Relation to the cited literature (what this package is NOT)

The cited papers describe a strictly **richer** model than this package:

- **Maggiar & Sadighian (2017)** — *Joint Inventory and Revenue Management with Removal Decisions*
  (Amazon.com working paper, SSRN 3018984). Their MDP also has a **pricing/markdown** decision and is
  solved under **backlogging** (stockouts satisfied in-period at cost `h- = c + k > c`), with
  **additive price-dependent Gamma demand** `D_t(p) = d_t(p) + e_t`. Their optimal policy is an
  **"interval-stock list-PRICES policy"** (Theorem 3.4): two stock levels `(x*, xbar*)` — order up to
  `x*` below it, remove down to `xbar*` above it, do nothing in between — *plus* a price/demand
  decision. Their only numerical example (Section 7, Table 1: p0=90, c=75, s=30, l=5, h+=2, k=15.5,
  elasticity -2; 40 periods; gamma=0.9984) reports an NPV surface (axis ~84000), inseparable from the
  pricing dimension. (Citation independently re-verified against the paper PDF, 2026-05-31.)

- **Maggiar, Andaz, Bagaria, Eisenach, Foster, Gottesman & Perrault-Joncas (2025)** —
  *Structure-Informed Deep Reinforcement Learning for Inventory Management* (NeurIPS 2025;
  arXiv:2507.22040; OpenReview `asKybwTGUt`). Section 4.6 lists joint procurement-removal (inventory
  with returns) as one DRL benchmark family, cites Maggiar & Sadighian (2017), and reports
  (Section 4.6.4) **qualitatively only** that the agent learns the **interval-stock** structure
  (Figure 23); it explicitly does **not** report the average expected reward for this family, so it
  exposes **no published cost row** for the procurement-removal problem. (Citation independently
  re-verified against the arXiv HTML, 2026-05-31.)

This package keeps the **interval-stock procurement/removal structure** and the returnable-quota
state but strips away pricing/markdown, uses lost-sales instead of backlog, and Poisson demand. It
is therefore a **repo-native control-only slice**, not the published model, and there is **no public
exact cost row to reproduce**.

## Faithful pricing-coupled model (added — the literature target)

Alongside the control-only slice above, a **faithful** environment now adds the PRICING / MARKDOWN
decision that the paper's reward structure (Eq. 4) requires, so the model matches the paper rather
than a reduction.

Files:

- `joint_pricing_removal_env.rs` — faithful state `z = (x, y)` (inventory level, returnable level),
  the joint decision (target demand `d` = price/markdown, and signed net flow `q`: remove if `q > 0`,
  purchase if `q < 0`), and the Eq. 4 period reward under backlogging with the paper's backorder
  convention `h- = c + k`. Includes the terminal value `V_T(x,y) = s*min(x,y) + l*max(x-y,0)`.
- `price_dependent_gamma_demand.rs` — the additive log-linear price-dependent demand
  `d_t(p) = mu_t exp(-beta(p-p0))` with inverse `p(d)`, plus the Gamma(mean mu_t, CV=1) noise and its
  K equally-likely quantile discretization (paper Sections 7.1.1 and 6.2.1).
- `joint_pricing_removal_dp.rs` — the executing finite-horizon DP that solves the value function
  `V_t(x,y)` (the NPV surface), recovers the optimal decisions, and exposes them for verification.

Literature instance and anchors live in `literature/references.rs`:

- `MAGGIAR_SADIGHIAN_2017_FAITHFUL_INSTANCE` — Table 1 parameters
  (`p0=90, c=75, s=30, l=5, h+=2, k=15.5, E=-2`, 40 periods, `gamma=0.9984`, 99 demand quantiles) and
  the reported NPV-surface peak `~84000` (Figure 7, t=24).
- `FAITHFUL_VERIFICATION_INSTANCE` — the same faithful dynamics shrunk to a coarse `(x,y)` grid so the
  DP solves exactly inside `cargo test`.

### What is reproduced (executing tests in `verification/tests.rs`)

- the price/demand log-linear map and elasticity relation,
- the Gamma(CV=1) noise quantiles (centred, variance ~ mu^2),
- the Eq. 4 single-period reward by hand, the `h- = c + k` backorder convention, and the
  return-then-liquidate terminal value,
- **the paper's PROVEN optimal-policy structure (Lemma 3.1 / Section 3.2 / 7.2.1)**: markdown/target
  demand nondecreasing in inventory and nonincreasing in returnable level, returns nondecreasing in
  inventory, purchases nonincreasing in inventory, and value-function supermodularity (L♮-concavity),
- the Table-1 NPV-surface magnitude: the env-computed peak at `t=24` brackets the reported `~84000`.

### What is NOT exactly reproduced (honest limitation)

The paper specifies the per-period mean-demand profile `mu_t` **only graphically** (Figure 6:
baseline ~50, peak ~500 near period 20). The headline output is a NPV **surface** at fixed `t=24`
whose top contour is `~84000`, not a single tabulated cost. Because `mu_t` and the conditional
plotting window are graphical, the exact `84000` is not reproducible to tight tolerance. The faithful
DP reproduces the paper's exact PROVEN structural properties (independent of the `mu_t` shape) and the
NPV magnitude band, so `literature_verified` stays **false** for the NPV figure; the structural
monotonicity / supermodularity reproduction is the executing literature-grounded correctness anchor.
The legacy control-only files are retained for the existing learned-policy pipeline.

## Verification status

Honest, itemized (no published number is reproduced, so this is **not** literature-verified):

- `literature_verified`: **no**. Both cited papers were independently re-verified during the
  2026-05-31 audit (paper PDF and arXiv:2507.22040 HTML); the citations are correct and neither paper
  exposes a public procurement-removal cost row (2017 numbers are pricing-coupled; 2025 reports the
  returns family qualitatively only). See `literature/README.md`.
- **self-consistent-only** (no public anchor): the reduced finite-horizon verifier
  (`finite_horizon_dp.rs`, `verification/tests.rs`) is validated against the repo's own exact DP, and
  that DP was **independently reproduced in pure Python to machine precision** (optimal discounted
  cost `31.7802611137`, diff `0.00e+00`) as part of this audit. This proves the env/DP are correctly
  implemented; it is not a literature claim.
- **faithful-to-structure**: the env matches the *structure* of Maggiar & Sadighian (2017)
  (interval-stock policy, return-before-liquidate / Corollary 1, fixed-returnability cap / Section
  3.2, salvage form / Assumption 4, cost-ordering / Assumption 2) but omits the pricing dimension the
  published numbers depend on, so no published number can anchor it.
- root cause of "not verified": structural reduction, not a bug.

## Instance set and benchmark

Two instances are benchmarked (see `literature/references.rs` and
`scripts/procurement_removal_inventory/benchmark_procurement_removal.py`):

- `PRIMARY_REFERENCE_INSTANCE` (`maggiar2017_style_fixed_returnability`): demand mean 4 over 16
  periods from 5 units. Demand drains inventory faster than it accumulates, so the system rarely
  overstocks and the **removal lever is essentially inactive**: the best constant interval-stock is
  `(order_up_to=6, remove_down_to=6)` — the removal level collapses onto the order level.
- `REMOVAL_ACTIVE_REFERENCE_INSTANCE` (`removal_active_returnability`): high initial inventory (12,
  of which 8 returnable), demand mean 3, holding cost 1.0. The system starts overstocked, so removing
  excess is worthwhile and the `remove_down_to` threshold **binds**: best constant interval-stock is
  `(order_up_to=4, remove_down_to=9)`, which beats both never-remove and aggressive-remove.

Benchmark results (mean discounted cost over 4096 held-out seeds, lower is better;
soft-tree = CMA-ES-trained depth-2 oblique linear-leaf policy, 80 generations, population 24;
recorded in `outputs/procurement_removal_inventory/benchmark_2026-05-31.json`):

| Instance | best interval_stock | best returnability_buffer | soft_tree |
| --- | ---: | ---: | ---: |
| primary (removal inactive) | 358.107 `(6,6)` | 358.107 `(6,6,0)` | 358.218 |
| removal_active (removal binds) | 244.117 `(4,9)` | 244.117 `(4,9,0)` | 251.727 |

Reading: on the primary instance the soft-tree essentially **ties** the best tuned interval-stock
(0.03% behind) — it recovers the interval-stock structure that the literature says is optimal for
this family. On the harder removal-active instance the tuned constant interval-stock is the strong
comparator and the depth-2 soft-tree is 3.1% behind at this small training budget (a larger budget /
deeper tree is the obvious next step). The reduced exact DP (separate small verifier instance)
dominates both heuristics, as it must.

## How to run

```
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py --train \
    --eval_seeds 4096 --generations 80 \
    --output_json outputs/procurement_removal_inventory/benchmark.json
```

(Drop `--train` for heuristics + exact-DP only.) This script depends only on the installed
`invman_rust` and `invman.cmaes`; it does **not** import the removed `invman.policies.soft_tree`.

## State interface

- `env.rs` exposes raw state quantities only
- the soft-tree benchmark uses an explicit policy-side feature map in `rollout.rs` (7 features,
  normalization is policy-owned, not hidden in the env)
- normalization / derived ratios must not be hidden inside the environment layer
