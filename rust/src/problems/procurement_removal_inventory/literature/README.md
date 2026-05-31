# Literature

Literature anchors for `procurement_removal_inventory`, with the exact claims verified against the
source PDFs during the 2026-05-31 audit.

## Maggiar & Sadighian (2017) — Joint Inventory and Revenue Management with Removal Decisions

<https://assets.amazon.science/7b/48/bc8c1c21450b9dac198e1f4ed13a/joint-inventory-and-revenue-management-with-removal-decisions.pdf>

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

<https://openreview.net/pdf?id=asKybwTGUt>

- Includes "Joint Procurement-Removal" as one DRL benchmark family and cites Maggiar & Sadighian
  (2017) ([13]) for it.
- Reports **only qualitatively** that "the DRL agent successfully learns interval-stock policies for
  inventory management with returns." There is **no published procurement-removal cost / reward /
  optimality-gap number** in the paper.

## Consequence for this package

- The executable package strips pricing/markdown, uses **lost sales** with a flat per-unit shortage
  cost, and **Poisson** demand. It keeps the interval-stock procurement/removal structure, the
  returnable-quota state, the return-before-liquidate rule, the salvage form, and the cost ordering.
- Because the published numbers are pricing-coupled (2017) or absent (2025), **no public exact cost
  row verifies this reduced package**. `literature_verified = false` is the honest status, and it is
  a structural reduction, not a defect.

## Source of truth in code

Use `literature/references.rs` for:

- `PRIMARY_REFERENCE_INSTANCE` — carried repo-native primary instance (removal lever inactive)
- `REMOVAL_ACTIVE_REFERENCE_INSTANCE` — repo-native instance where the removal lever binds (added in
  the 2026-05-31 audit so the benchmark exercises the distinguishing procurement-vs-removal feature)
- `VERIFICATION_PROBLEM_INSTANCE` — reduced exact-DP verifier instance
- `MAGGIAR_2017_REFERENCE`, `MAGGIAR_2025_REFERENCE` — carried benchmark-policy names and notes
