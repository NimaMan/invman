# Literature

Two distinct papers; keep them separate.

Headline paper (full truck-dispatch model):

- Giannoccaro and Pontrandolfo (2010), *A Reinforcement Learning Approach for Inventory Replenishment
  in Vendor-Managed Inventory Systems With Consignment Inventory*, DOI 10.1080/10429247.2010.11431878

Verified analytical anchor (the worked newsvendor case):

- Sui, Gosavi, and Lin (2010), *A reinforcement learning approach for inventory replenishment in
  vendor-managed inventory systems with consignment inventory*, Engineering Management Journal,
  22(4): 44-53
- public worked case study (Gosavi 2020, derived from the above): `vmi_case_study.pdf`
- author MATLAB code for the same case: `vmi_newsvendor.m`

What is carried here:

- the 8 designed-experiment case definitions for the headline paper (executable, but NOT benchmark
  anchors — see below)
- the public single-retailer worked newsvendor calculation (the verified anchor)
- the repo-native reduced single-retailer primary instance and exact verifier instance

Verification confirmation (2026-05-31):

- the `vmi_case_study.pdf` was fetched and converted to text; every displayed quantity in the worked
  example was checked against `verification/newsvendor_case.rs`. All match:
  `mu = 0.375`, `sigma^2 = 0.5833`, `mu_C = 40`, `sigma_C^2 = 50`, cycle-demand mean `15`,
  cycle-demand variance `30.36`, mean-demand heuristic `S = 15`, six-sigma `S = 31.53`,
  newsvendor `S = 26.96`. The env's newsvendor `S = 26.99` differs only because the PDF truncates
  `k = Phi^-1(0.98) = 2.17` while the env uses full-precision `k = 2.176`; the verification tolerance
  of `0.05` covers this 0.03 rounding gap.
- the moment formulas in the env are exactly the paper's: Wald's-equation compound-Poisson demand
  moments, the random-sum cycle-demand variance `mu_C * sigma^2 + mu^2 * sigma_C^2`, and the
  classical newsvendor critical ratio `p / (p + h)`.

Important distinction:

- the root Rust environment also implements the paper’s continuous-time multi-retailer truck-dispatch
  structure (`env::step_paper_state`)
- but one parameter family remains partially implicit in the paper, especially the numerical high/low
  demand-signal semantics
- the paper’s `T + L` newsvendor derivation is consistent with same-cycle dispatch and same-cycle
  retailer arrival for the current decision; the lagged-arrival interpretation was tested and
  rejected because it worsened case 1
- the paper’s reported benchmark profit uses retailer holding, retailer stockout, revenue, and
  truck operating cost, not DC-side holding or reorder terms

So the public worked newsvendor case is literature-verified. The full 8-case paper profit table is
dropped from the benchmark layer because the remaining row-level gap is statistically meaningful and
the demand-signal process is not public enough to resolve it. Policy benchmarking is therefore done
on the reduced single-retailer slice; see the root README's Benchmark section.
