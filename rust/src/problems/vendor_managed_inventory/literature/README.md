# Literature

Primary anchor:

- Giannoccaro and Pontrandolfo (2010), reinforcement learning for vendor-managed inventory with consignment inventory

Public companion material:

- Gosavi worked newsvendor case study
- Gosavi MATLAB code for the same case

What is carried here:

- the paper-level benchmark rows for the 8 designed experiments
- the public single-retailer worked newsvendor calculation
- the repo-native reduced single-retailer primary instance and exact verifier instance

Important distinction:

- the root Rust environment now follows the paper’s continuous-time multi-retailer truck-dispatch structure
- but one parameter family remains partially implicit in the paper, especially the numerical high/low demand-signal semantics
- the paper’s `T + L` newsvendor derivation is consistent with same-cycle dispatch and same-cycle
  retailer arrival for the current decision; the lagged-arrival interpretation was tested and
  rejected because it worsened case 1
- the paper’s reported benchmark profit uses retailer holding, retailer stockout, revenue, and
  truck operating cost, not DC-side holding or reorder terms

So the public worked newsvendor case is literature-verifiable. The full 8-case paper profit table is
dropped from the benchmark layer because the remaining row-level gap is statistically meaningful and
the demand-signal process is not public enough to resolve it.
