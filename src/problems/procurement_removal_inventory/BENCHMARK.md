# procurement_removal_inventory — benchmark card

**One-line MDP:** state = (period, on-hand inventory, returnable inventory); action = (purchase quantity, removal quantity); one-period cost = purchase + holding + lost-sales shortage − (return + liquidation credits); objective = minimize discounted finite-horizon expected cost (γ = 0.99).

**Status:** `faithful_unverified` (env faithful-to-STRUCTURE of Maggiar & Sadighian 2017; what re-ran is repo-native self-consistency; **NO published number exists** — `no_published_number`). **Paper:** no dedicated benchmark section; Maggiar et al. (2025) is cited only in the discussion/related-work of `learning_inventory_control_policies_es.tex` (around line 3697, `\citet{maggiar2025structure}`) as a structure-informed-DRL touchstone, not as a reproduced row.

## Problem formulation

Single-item, finite-horizon, lost-sales system with a vendor-returns channel. Per-period event order in `env.rs::step_state`:

1. **Purchase arrives immediately:** `inventory += q`; `min(q, returnable_purchase_cap)` of the purchased units enter the returnable pool (the fixed-returnability contract, Maggiar & Sadighian 2017 §3.2 — a per-period cap on returnable purchases).
2. **Removal:** `removal_quantity` units leave; returnable units are returned first, the remainder is liquidated (return-before-liquidate, the paper's Corollary 1).
3. **Demand realizes:** lost sales — `sales = min(demand, on_hand)`, unmet demand is lost and charged `shortage_cost_per_unit`.
4. **Holding cost** is charged on the **ending** on-hand inventory.

One-period cost (`reward = -period_cost`):

```
period_cost = purchase_cost + holding_cost + shortage_cost
              - (return_value * returned_units + liquidation_value * liquidated_units)
```

At the horizon a terminal salvage credit `VT(x,y) = s*min(x,y) + l*max(x-y,0)` is applied (Maggiar & Sadighian 2017, Assumption 4). Cost ordering enforced in `env.rs::validate_costs` matches the paper's Assumption 2: `purchase_cost > return_value` (2.ii) and `return_value >= liquidation_value` (2.iii). State invariant: `returnable_inventory <= inventory_level`.

**Long-run objective:** minimize the discounted sum of one-period costs minus the terminal salvage credit over the finite horizon, with discount γ = 0.99.

## Reference instances

| instance | dimensions covered | key params | literature_verified flag |
| --- | --- | --- | --- |
| `maggiar2017_style_fixed_returnability` (PRIMARY) | lost_sales / poisson / mean 4.0 / 16 periods / returnable cap 2 / **removal inactive** / γ=0.99 | init_inv 5, init_returnable 3, c=6.0, return=4.0, liq=1.0, h=0.5, p=9.0, max_purchase=6, max_removal=6; best interval-stock (6,6) | **false** (repo_native_instance_not_verified_against_literature) |
| `removal_active_returnability` | lost_sales / poisson / mean 3.0 / 16 periods / **removal channel binds** / γ=0.99 | init_inv 12, init_returnable 8, cap 2, c=6.0, return=4.0, liq=1.0, h=1.0, p=9.0, max_purchase=6, max_removal=8; best interval-stock (4,9) | **false** (repo_native_removal_active_instance) |
| `VERIFICATION_PROBLEM_INSTANCE` (reduced exact-DP) | lost_sales / discrete support [0,1,2,3] (p=[.2,.3,.3,.2]) / 5 periods / returnable cap 1 / γ=0.99 / **exact-DP solvable** | init_inv 2, init_returnable 1, c=5.0, return=3.0, liq=1.0, h=0.5, p=7.0, max_purchase=4, max_removal=4 | **false** (repo_exact_solver_not_verified_against_literature) |

All three instances are repo-native: the cited papers expose no public per-instance procurement-removal control-only cost.

## Baselines

- **Heuristics:** `interval_stock (order_up_to, remove_down_to)` — the structural-optimum form per Maggiar & Sadighian Theorem 3.4 ("interval-stock list-prices policy"), grid-tuned; and `returnability_buffer_interval_stock (order_up_to, remove_down_to, returnable_buffer)`. Searched by grid over the `(order_up_to, remove_down_to[, buffer])` levels.
- **Exact / optimal:** `finite_horizon_dp.rs::solve_optimal_policy` — exact bounded backward-induction DP, **ONLY** on the small discrete-support `VERIFICATION_PROBLEM_INSTANCE` (periods=5). Repo-native optimum, **NOT published**. The two benchmark instances (periods=16, Poisson) are **not** solved exactly.
- **Published comparators (CONTEXT only):** **NONE usable as a cost row.** Maggiar & Sadighian 2017's only numerical example (§7, Table 1: p0=90, c=75, s=30, l=5, h+=2, k=15.5, elasticity −2; 40 periods; γ=0.9984) is a pricing-coupled NPV surface (axis ~84000), inseparable from the pricing dimension this repo strips. Maggiar et al. 2025 (NeurIPS, arXiv:2507.22040) reports the returns family **qualitatively only** (Fig 23 — agent learns the interval-stock structure) and explicitly does not report average expected reward for it. Both are cross-model / no-cost-row CONTEXT, never a "beats" target.

## Verification

- **Published number:** none — neither cited paper exposes a public procurement-removal control-only cost row.
- **Re-run reproduced (repo-native self-consistency only):**
  - exact-DP `optimal_discounted_cost = 31.78026111369698` (README claim `31.7802611137`, match to 1e-10; independently re-implemented in pure Python to machine precision, abs diff `0.00e+00`) via `r.procurement_removal_inventory_exact_dp_summary()`.
  - primary `interval_stock (6,6) = 358.1067286254911` reproduced exactly; removal_active `interval_stock (4,9) = 244.11666203081566` reproduced exactly, via `procurement_removal_inventory_simulate_policy('interval_stock', [6,6]/[4,9], seeds=range(500000,504096), discount=0.99)`.
  - worked-transition `period_cost = 10.5` via `procurement_removal_inventory_step(...)` (assertions pass).
  - On the verifier instance the exact DP (31.78026) dominates both heuristics (`interval_stock` 34.164, `returnability_buffer` 38.766), as it must.
- **Verdict:** `faithful_unverified` / `no_published_number`. This is a **deliberate structural reduction** of Maggiar & Sadighian 2017 (pricing/markdown stripped, lost-sales instead of backlog, Poisson demand), not a bug. The env is faithful to the published *structure* (interval-stock policy, return-before-liquidate / Corollary 1, fixed-returnability cap / §3.2, salvage / Assumption 4, cost ordering / Assumption 2), but the pricing dimension the published numbers depend on is omitted, so **no published number can anchor it**. The DP self-consistency re-run proves correct implementation only — it is **not** a literature claim. This is a standing verification debt: no public anchor exists.

## Results (learned policy)

soft_tree = CMA-ES-trained depth-2 oblique linear-leaf policy (80 generations, population 24), evaluated over 4096 held-out seeds; recorded in `outputs/procurement_removal_inventory/benchmark_2026-05-31.json`.

| Instance | best interval_stock | best returnability_buffer | soft_tree |
| --- | ---: | ---: | ---: |
| primary (removal inactive) | 358.107 `(6,6)` | 358.107 `(6,6,0)` | 358.218 |
| removal_active (removal binds) | 244.117 `(4,9)` | 244.117 `(4,9,0)` | 251.727 |

- **primary:** soft_tree 358.218 essentially **ties** the best tuned interval-stock 358.107 (0.03% behind). **NOT a beat.** **single_seed — NOT yet seed-robust** (manifest `at_risk: true`).
- **removal_active:** soft_tree 251.727 is **3.1% BEHIND** interval-stock 244.117 — the heuristic wins. **NOT a beat.** **single_seed — NOT yet seed-robust** (manifest `at_risk: true`).
- **verifier (exact DP):** optimal 31.7802611137 dominates both heuristics (34.164, 38.766). Re-run, holds (`at_risk: false`; not a learned-policy claim).

No claim here beats the same-protocol gate. The two soft-tree rows are single-seed and must be treated as not seed-robust per the seed-robust reporting standard; a larger budget / deeper tree is the obvious next step on the removal-active instance.

## Reproduce

```bash
# exact-DP verifier (optimal discounted cost + first action)
python -c "import invman_rust as r; s=r.procurement_removal_inventory_exact_dp_summary(); print(s['optimal_discounted_cost'], s['optimal_first_action'])"

# heuristics + exact-DP only (no training)
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py

# full benchmark with soft-tree training (4096 held-out eval seeds, 80 generations)
python scripts/procurement_removal_inventory/benchmark_procurement_removal.py --train --eval_seeds 4096 --generations 80 \
    --output_json outputs/procurement_removal_inventory/benchmark.json

# worked single-transition assertion (period_cost = 10.5)
python -c "import invman_rust as r; print(r.procurement_removal_inventory_step(4,2,3,2,4,2,6.0,4.0,1.0,0.5,9.0))"
```

## Pointers & caveats

- **code:** `src/problems/procurement_removal_inventory/` — `env.rs` (MDP / step), `finite_horizon_dp.rs` (exact DP verifier), `literature/references.rs` (instance params + reference metadata), `literature/README.md`, `verification/tests.rs` + `verification/README.md`, `heuristics/`, `rollout.rs` (policy-side 7-feature map, normalization is policy-owned), `bindings.rs`.
- **scripts:** `scripts/procurement_removal_inventory/` — `benchmark_procurement_removal.py`, `validate_against_exact_dp.py`, `train_soft_tree_reference.py`, `common.py`.
- **autoresearch:** no `policy_search/programs/program_procurement_removal_inventory.md` exists for this system.
- **Honest caveats:**
  - **No published cost row exists** — `literature_verified = false`, `no_published_number`. Verification re-ran is repo-native self-consistency (exact DP to machine precision + heuristic reproduction), not a literature anchor. This is the standing debt.
  - 2017 published numbers are **pricing-coupled NPV** (~84000) and 2025 reports the returns family **qualitatively only** — both are CONTEXT, never a "beats" comparator.
  - The repo is a **control-only reduction** of Maggiar & Sadighian 2017: pricing/markdown removed, **lost-sales** instead of backlog, Poisson demand. Faithful to structure, not to the published numerical setting.
  - Exact DP is available **only** on the reduced 5-period discrete-support verifier instance; the two 16-period Poisson benchmark instances have **no exact optimum** to gap against.
  - Both soft-tree result rows are **single-seed, NOT seed-robust**, and neither beats its same-protocol gate.
  - The existing `README.md` in this folder is consistent with the manifest/ledger (same instances, same numbers, same honest `literature_verified = false` status); this card is additive, not a correction.
