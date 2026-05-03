# Vendor Managed Inventory

Rust-first problem home for `vendor_managed_inventory`.

## Formulation

Original literature family:

- one vendor-controlled DC serving multiple retailers under a consignment inventory contract
- two products in the published numerical study
- retailer demand modeled as a compound Poisson process
- random cycle times driven by transport and service times
- truck-capacity dispatch decisions at the start of each cycle
- a newsvendor-based allocation heuristic to split shipped inventory across retailer-product pairs
- DC replenishment managed with a `(Q,R)` rule

Current Rust environment:

- continuous-time, cycle-based multi-retailer truck-dispatch simulator
- 10 retailers and 2 products in the carried Giannoccaro benchmark family
- compound-Poisson retailer demand with discrete-uniform demand sizes
- random route cycle times and retailer-specific lead times
- truck-count dispatch action with a newsvendor-based allocation rule
- DC `(Q,R)` replenishment with random manufacturer lead times

The older reduced single-retailer finite-horizon slice is still kept only as verification support.

## Literature Anchor

Primary paper:

- Giannoccaro and Pontrandolfo (2010), *A Reinforcement Learning Approach for Inventory Replenishment in Vendor-Managed Inventory Systems With Consignment Inventory*
- DOI: <https://doi.org/10.1080/10429247.2010.11431878>

Public companion material:

- worked newsvendor case study: <https://web.mst.edu/_disabled/gosavia/vmi_case_study.pdf>
- author MATLAB code for that case: <https://web.mst.edu/_disabled/gosavia/vmi_newsvendor.m>

Published paper experiment rows:

- the paper reports an 8-case table with newsvendor and RL profits
- those profit rows are not carried as benchmark assertions because the public text does not define
  the high/low demand-signal process tightly enough to reproduce the rows

## Current Status

- literature-verified: yes for the public worked newsvendor case-study calculation only
- literature-verified: no for the full Giannoccaro truck-dispatch benchmark
- repo-exact verified: yes on the reduced single-retailer verifier

Current benchmark status on the paper-facing simulator:

- the public analytical newsvendor worked case is reproduced exactly enough
- the 8-case truck-dispatch case definitions are executable in Rust, but their published profit
  rows are dropped from the benchmark layer
- the paper timing audit favors same-cycle dispatch with same-cycle retailer arrival for the current
  cycle’s trucks; the alternative next-cycle arrival interpretation moved case 1 farther away from
  the paper row and was rejected
- the paper objective audit also shows that the published profit excludes DC holding, DC shortage,
  and DC reorder costs; once that objective is used, reproduced case 1 newsvendor profit moves into
  the right range at about `16.4` against the published `15.41`
- the remaining gap is still statistically meaningful, so the full paper table is not used for
  verification or paper comparisons

## Structure

- [literature/README.md](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/literature/README.md)
- [verification/README.md](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/verification/README.md)
- [experiments/README.md](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/experiments/README.md)
- [practical/README.md](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/practical/README.md)

Code layout:

- root env / rollout / heuristics: paper-first continuous-time VMI truck-dispatch environment
- [references.rs](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/literature/references.rs): literature rows and problem instances
- [newsvendor_case.rs](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/verification/newsvendor_case.rs): literature-backed analytical verification helper
- [tests.rs](/home/nima/code/ml/invman/rust/src/problems/vendor_managed_inventory/verification/tests.rs): executable verification assertions
