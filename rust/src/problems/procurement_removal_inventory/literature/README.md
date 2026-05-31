# Literature

Literature anchors for `procurement_removal_inventory`. The two cited papers and every structural
and numerical claim below were independently re-verified against the original sources during the
2026-05-31 librarian audit (paper PDF for the 2017 reference; arXiv HTML 2507.22040 for the 2025
reference). Both citations are real and the metadata is correct.

## Maggiar & Sadighian (2017) — Joint Inventory and Revenue Management with Removal Decisions

Alvaro Maggiar and Ali Sadighian (Amazon.com). Working paper, August 14, 2017. SSRN abstract 3018984;
PDF mirrored at amazon.science.

<https://assets.amazon.science/7b/48/bc8c1c21450b9dac198e1f4ed13a/joint-inventory-and-revenue-management-with-removal-decisions.pdf>

<https://ssrn.com/abstract=3018984>

What the paper actually models (verified from the PDF):

- State `z_t = (x_t, y_t)`: inventory level `x_t` and **returnable** inventory level `y_t`.
- Three decisions per period: (i) returnable units to purchase/return `q_r`, (ii) non-returnable
  units to purchase/liquidate `q_nr`, (iii) a **target mean demand / price** (a pricing/markdown
  decision). Pricing is a core decision, not optional.
- Demand is **additive and price-dependent**: `D_t(p) = d_t(p) + e_t`, with Gamma-distributed noise
  (coefficient of variation 1 in the numerical example).
- Stockouts are handled by **in-period fulfilment at a cost above the purchase cost**
  (`h- = c + k`), interpolating between full backlog and lost sales; the primary analysis is the
  backlogging case (Remark 1).
- Cost ordering assumptions: `c > s` (purchase cost above return value, Assumption 2.ii) and `l < s`
  (liquidation below return value, Assumption 2.iii). Corollary 1: never liquidate a unit that could
  be returned; never buy non-returnable when a returnable unit can be bought.
- Fixed returnability (Section 3.2): a per-period cap on the number of returnable units that can be
  purchased.
- **Optimal policy structure (Theorem 3.4): an "interval-stock list-prices policy"** — two stock
  levels `(x*, xbar*)` with `x* <= xbar*`: below `x*` order up to `x*`; above `xbar*` remove down to
  `xbar*`; in between do nothing — *plus* a price/expected-demand decision.
- Terminal value example (Assumption 4): `VT(x,y) = s*min(x,y) + l*max(x-y,0)`.
- Numerical example (Section 7, Table 1): p0=90, c=75, s=30, l=5, h+=2, k=15.5, elasticity -2;
  40 periods; discount 0.9984; ~84000 NPV surface. The reported result is the **pricing-coupled NPV
  surface / decision contours**, not a standalone inventory-control cost row.

## Maggiar et al. (2025) — Structure-Informed Deep RL for Inventory Management

Alvaro Maggiar, Sohrab Andaz, Akhil Bagaria, Carson Eisenach, Dean Foster, Omer Gottesman, Dominique
Perrault-Joncas. NeurIPS 2025; arXiv:2507.22040; OpenReview id `asKybwTGUt`. (Alvaro Maggiar is first
author, so the short cite "Maggiar et al. (2025)" is correct.)

<https://openreview.net/pdf?id=asKybwTGUt>

<https://arxiv.org/abs/2507.22040>

- Section 4.6 ("Multi-Period Inventory Management with Returns") is the joint procurement-removal
  benchmark family; it cites Maggiar & Sadighian (2017) and describes the optimal **interval-stock**
  policy (order-up-to `s_lower`, remove-down-to `s_upper`), illustrated in its Figure 22.
- Section 4.6.4 ("Results") reports **only qualitatively** that the DRL agent "learns the right
  structural properties of the optimal policy" (Figure 23). It explicitly states it does **not**
  present the measured average expected reward for this family (its steady state coincides with the
  basic multi-period problem of Section 4.2), so there is **no published procurement-removal cost /
  reward / optimality-gap number** in the paper.

## Consequence for this package — verifiability status (honest)

- The executable package strips pricing/markdown, uses **lost sales** with a flat per-unit shortage
  cost, and **Poisson** demand. It keeps the interval-stock procurement/removal structure, the
  returnable-quota state, the return-before-liquidate rule, the salvage form, and the cost ordering.
- Because the published numbers are pricing-coupled (2017, Table 1 / NPV surface ~84000) or absent
  for the returns family (2025, qualitative only), **no public exact cost row verifies this reduced
  package**.

Itemized status of each block (do not overclaim):

- **Literature-verified (env reproduces a published number with a solver):** none. There is no public
  procurement-removal cost row to reproduce.
- **Faithful-to-structure but no published anchor:** the env structure (interval-stock policy,
  return-before-liquidate per Corollary 1, fixed-returnability cap per Section 3.2, terminal salvage
  `s*min(x,y)+l*max(x-y,0)` per Assumption 4, cost ordering c>s and l<=s per Assumption 2) faithfully
  matches Maggiar & Sadighian (2017), but the pricing dimension that the published numbers depend on
  is omitted, so no published number can anchor it.
- **Self-consistent-only (validated against the repo's own exact solver, no public anchor):** the
  reduced finite-horizon DP on `VERIFICATION_PROBLEM_INSTANCE`. The Rust exact DP was independently
  re-implemented in pure Python during this audit and the two agree to machine precision (optimal
  discounted cost `31.7802611137`, absolute difference `0.00e+00`; see `../verification/README.md`).
  This confirms the env/DP are correctly implemented; it is **not** a literature claim.
- **Table-only (published numbers stored but not re-derived):** none. No published numeric values are
  stored as benchmark anchors (`reported_numbers_available = false`,
  `numbers_anchor_repo_assertions = false` for both references).
- **Repo-native benchmark instances (not literature rows):** `PRIMARY_REFERENCE_INSTANCE` and
  `REMOVAL_ACTIVE_REFERENCE_INSTANCE`. Their `(order_up_to, remove_down_to)` levels and the benchmark
  costs in the top-level README are repo-native, self-consistent results, not published numbers.

`literature_verified = false` is therefore the correct status: it is a deliberate structural
reduction, not a defect.

## Source of truth in code

Use `literature/references.rs` for:

- `PRIMARY_REFERENCE_INSTANCE` — carried repo-native primary instance (removal lever inactive)
- `REMOVAL_ACTIVE_REFERENCE_INSTANCE` — repo-native instance where the removal lever binds (added in
  the 2026-05-31 audit so the benchmark exercises the distinguishing procurement-vs-removal feature)
- `VERIFICATION_PROBLEM_INSTANCE` — reduced exact-DP verifier instance
- `MAGGIAR_2017_REFERENCE`, `MAGGIAR_2025_REFERENCE` — carried benchmark-policy names and notes
