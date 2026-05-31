# Literature

## CITATION CORRECTION (2026-05-31, librarian audit)

A previous version of this folder cited the headline model as **"Giannoccaro and Pontrandolfo
(2010)"** with DOI `10.1080/10429247.2010.11431878`. That attribution is **wrong**. Independent
verification (Crossref + Taylor & Francis) shows that DOI and title belong to:

- **Sui, Z., Gosavi, A., and Lin, L. (2010)**, *A Reinforcement Learning Approach for Inventory
  Replenishment in Vendor-Managed Inventory Systems With Consignment Inventory*,
  *Engineering Management Journal*, **22(4): 44-53**, DOI `10.1080/10429247.2010.11431878`.
  - Verified: <https://api.crossref.org/works/10.1080/10429247.2010.11431878> (authors: Zheng Sui,
    Abhijit Gosavi, Li Lin; EMJ 22(4):44-53; 2010), and
    <https://www.tandfonline.com/doi/abs/10.1080/10429247.2010.11431878>.

There is **no** "Giannoccaro and Pontrandolfo (2010)" VMI paper. Giannoccaro & Pontrandolfo did
publish *Inventory management in supply chains: a reinforcement learning approach*,
*International Journal of Production Economics*, **78(2): 153-161 (2002)**
(DOI `10.1016/S0925-5273(00)00156-0`), but that is a **different** model (a generic three-stage
serial supply chain, not a VMI / consignment / truck-dispatch system) and is **not** the source used
here.

The Rust identifiers `GIANNOCCARO_2010_REFERENCE`, `GIANNOCCARO_2010_NEWSVENDOR_WORKED_CASE`,
`GIANNOCCARO_2010_CASE_DEFINITIONS`, and `build_giannoccaro_2010_case` still carry the wrong name. They
were left unrenamed in this pass because they are referenced from `bindings.rs`, `rollout.rs`, and
`verification/tests.rs`, and renaming them requires a Rust rebuild + binding regeneration (out of
scope for a string-only citation fix). Renaming them to `SUI_GOSAVI_LIN_2010_*` is a tracked blocker.

## The single source paper

- **Sui, Gosavi, and Lin (2010)** — see above. This one paper supplies BOTH the continuous-time
  multi-retailer truck-dispatch model AND the worked newsvendor allocation calculation.

## Verified analytical anchor (the worked newsvendor case)

The executable anchor is a public **instructor case study by Abhijit Gosavi** that re-works a single
retailer/product newsvendor example from Sui, Gosavi & Lin (2010):

- Gosavi, A. — *Case Study for Vendor-Managed Inventory (Based on Sui, Gosavi, & Lin, 2010)*,
  Missouri University of Science and Technology, dated September 7, 2010 (PDF footer year 2020).
  - PDF: <https://web.mst.edu/_disabled/gosavia/vmi_case_study.pdf>
  - MATLAB code link in the repo: `https://web.mst.edu/_disabled/gosavia/vmi_newsvendor.m`

Verification status of these URLs (2026-05-31): the `vmi_case_study.pdf` URL **loads** (663 KB) and was
fetched and read directly during this audit. The `vmi_newsvendor.m` link was **not** independently
confirmed to load in this pass (the directory index and the non-`_disabled` `~gosavia/` path 404'd);
treat the MATLAB-code URL as unverified.

## What is carried here

- the public single-retailer worked newsvendor calculation (the verified analytical anchor)
- the 8 designed-experiment case **definitions** for the truck-dispatch model — these are executable
  scaffolding, but are **NOT** benchmark anchors (no published profit row is reproduced)
- the repo-native reduced single-retailer primary instance and exact-verifier instance (repo-chosen
  parameters, **not** from any published table)

## Worked-case verification confirmation (2026-05-31)

The `vmi_case_study.pdf` was fetched and read; every displayed quantity in the worked example was
checked against `verification/newsvendor_case.rs`. All match the published PDF (page 4-5):

- customer Poisson rate `lambda = 0.25`, demand size `UNIF(1,2)`, `h = 0.06`, `p = 4.00`,
  cycle time `T in {30,40,50}` with probs `{0.25, 0.5, 0.25}`
- `mu = 0.375`, `sigma^2 = 0.5833`, `mu_C = 40`, `sigma_C^2 = 50`
- cycle-demand mean `15`, cycle-demand variance `23.33 + 7.03 = 30.36`
- mean-demand heuristic `S = 15`, six-sigma `S = 15 + 3*sqrt(30.36) = 31.53`
- newsvendor `F(S) = 4/(4+0.06) = 0.9852`, `k = Phi^-1(0.98) = 2.17`, `S = 26.96`

The env's newsvendor `S = 26.99` differs from the PDF's `26.96` only because the PDF truncates
`k = Phi^-1(0.98)` to `2.17` while the env uses the full-precision `k = 2.176`; the verification
tolerance of `0.05` covers this 0.03 rounding gap. The moment formulas in the env are exactly the
PDF's: Wald's-equation compound-Poisson demand moments, the random-sum cycle-demand variance
`mu_C * sigma^2 + mu^2 * sigma_C^2` (Nahmias 2001), and the classical newsvendor critical ratio
`p / (p + h)` (Askin & Goldberg 2002; Nahmias 2001).

This worked case is therefore **literature-verified** (reproduced from the public source, not merely
stored).

## Important distinction (the full truck-dispatch model is NOT verified)

- the root Rust environment also implements the paper's continuous-time multi-retailer truck-dispatch
  structure (`env::step_paper_state`)
- but one parameter family remains partially implicit in the public material, especially the numerical
  high/low demand-signal semantics
- the public worked case's `T + L` newsvendor derivation is consistent with same-cycle dispatch and
  same-cycle retailer arrival for the current decision; the lagged-arrival interpretation was tested
  and rejected because it worsened case 1
- the published benchmark profit uses retailer holding, retailer stockout, revenue, and truck
  operating cost, not DC-side holding or reorder terms

So: the public worked newsvendor case is **literature-verified**. The full 8-case paper profit table
is **dropped** from the benchmark layer because the row-level gap is statistically meaningful and the
demand-signal process is not public enough to resolve it (the original Sui/Gosavi/Lin dataset would be
needed). Policy benchmarking is therefore done on the repo-native reduced single-retailer slice (which
has **no published anchor** and is **self-consistent-only**, validated against the repo's own exact
finite-horizon DP); see the root README's Benchmark section.
