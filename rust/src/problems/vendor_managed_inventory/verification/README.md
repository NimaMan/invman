# Verification

This package has two verification scopes.

Literature-backed analytical verification:

- the public worked newsvendor case study posted by the paper’s author (Sui, Gosavi & Lin 2010 /
  Gosavi 2020 `vmi_case_study.pdf`)
- exact reproduction of the displayed mean/variance calculations and order-up-to levels
- confirmed on 2026-05-31 by fetching the source PDF and matching every displayed quantity to
  `newsvendor_case.rs` (see the line-by-line table in the root README; the only deviation is the
  newsvendor `S = 26.96` published vs `26.99` env, caused solely by the PDF truncating
  `k = Phi^-1(0.98) = 2.17`, which the `0.05` test tolerance absorbs)

Repo-native executable verification:

- a reduced single-retailer VMI instance with:

  - explicit DC stock
  - one-period retailer shipment pipeline
  - deterministic DC replenishment
  - lost-sales retailer demand

The executable assertions live in:

- `rust/src/problems/vendor_managed_inventory/verification/newsvendor_case.rs`
- `rust/src/problems/vendor_managed_inventory/verification/tests.rs`

Paper benchmark audit:

- the full 10-retailer, 2-product Giannoccaro benchmark is now executable via the root env/heuristic/rollout path
- but the reproduced newsvendor profits still do not match the published table closely enough to use
  those rows as verification anchors
- those full-table profit rows are therefore dropped from the benchmark layer; only the public worked
  newsvendor calculation remains literature-verified

Main unresolved paper-specific assumptions:

- the numerical meaning and transition law of the high/low demand-forecast update are not given explicitly in the public paper text
- the public benchmark description does not pin down the frozen-stage initialisation protocol tightly enough for exact replication

Timing conclusion from the audit:

- the paper’s own newsvendor derivation defines the protection horizon as `T + L`, where `L` is the
  current cycle’s retailer lead time and `T` is the cycle time to the next review
- that is consistent with same-cycle dispatch and same-cycle retailer arrival for the current
  decision, with coverage until the next cycle’s arrival
- a forced next-cycle retailer-arrival convention was tested and moved case 1 farther away from the
  published profit, so it is not kept in the executable paper path

Objective conclusion from the audit:

- the paper’s stated benchmark objective uses four terms only: retailer holding cost, retailer
  stockout penalty, sales revenue, and truck operating cost
- `hDC`, `pDC`, and `K` are listed in Exhibit 5 and still affect the DC replenishment dynamics, but
  they are not part of the reported paper profit metric
- including DC holding, DC shortage, or DC reorder costs in the paper benchmark objective pushes
  case 1 far away from the published row and is therefore not used in the literature-facing reward
