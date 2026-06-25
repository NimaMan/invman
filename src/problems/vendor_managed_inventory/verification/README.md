# Verification

This package has two verification scopes, neither of which is literature verification of a
peer-reviewed-paper number (see the honesty note at the bottom).

Instructor-case analytical reproduction:

- the Gosavi instructor teaching case study worked newsvendor example
- exact reproduction of the displayed mean/variance calculations and order-up-to levels
  (mu=0.375, sigma^2=0.5833, mu_cycle=15, sigma^2_cycle=30.36, MDH S=15, six-sigma S=31.53,
  newsvendor S=26.96)

Repo-native executable verification:

- a reduced single-retailer VMI instance with:
  - explicit DC stock
  - one-period retailer shipment pipeline
  - deterministic DC replenishment
  - lost-sales retailer demand
- correctness of mechanics, heuristic agreement, terminal salvage, and finite-horizon DP dominance

The executable assertions live in:

- `src/problems/vendor_managed_inventory/verification/newsvendor_case.rs`
- `src/problems/vendor_managed_inventory/verification/tests.rs`

Key tests:

- `newsvendor_worked_case_reproduces_gosavi_instructor_case_study`: re-runs the family's newsvendor
  solver and reproduces the Gosavi instructor-case worked example exactly
- `literature_verified_flags_are_honest`: drift guard that asserts the `references.rs` honesty flags
  stay `false` and that the source strings correctly attribute the paper to Sui/Gosavi/Lin (2010)
  and label the worked example as the Gosavi instructor teaching case study

Honesty note (per docs/rust/README.md "What counts as literature-verified"):

- NEITHER scope is literature verification of a peer-reviewed-paper number.
  - The Gosavi worked example is an INSTRUCTOR teaching handout, not the peer-reviewed paper; per the
    repo rule, an instructor/handout number is explicitly NOT literature verification.
  - The repo-native finite-horizon verifier proves self-consistency against our own DP, which is also
    NOT literature verification.
- The peer-reviewed Sui/Gosavi/Lin (2010) results table (pp. 44-53) is paywalled and not openly
  reproducible, so no number printed in the peer-reviewed paper is re-run.
- Therefore `references.rs` carries `literature_verified = false`.

Paper benchmark audit (why the full table is not usable):

- the repo's 10-retailer/2-product truck-dispatch case definitions are a structural interpretation,
  not a transcribed published table
- the high/low demand-forecast update is not given explicitly in any openly accessible source
- a prior audit found the reproduced case-1 newsvendor profit (~16.4) did not match a figure-only
  number (~15.41) closely enough to anchor verification, and even that 15.41 was read from a figure,
  not an openly available results table
- the published profit rows are therefore not used for verification or paper comparisons
