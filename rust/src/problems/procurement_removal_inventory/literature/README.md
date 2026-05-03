# Literature

Current literature anchors for `procurement_removal_inventory`:

- Maggiar and Sadighian 2017
- Maggiar et al. 2025

Repo interpretation:

- procurement and removal are modeled jointly
- returnable stock is part of the canonical state
- the current executable package strips away pricing and markdown decisions from the full
  Maggiar/Sadighian model, so it should be treated as a repo-native control slice

Use `literature/references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes

Status:

- Maggiar and Sadighian (2017) gives the full joint replenishment, pricing, and removal model with
  structural policy results and graphical numerical examples
- Maggiar et al. (2025) reports that learned policies recover interval-stock structure for
  procurement/removal, but does not expose an exact public cost row for this repo package
- this package is therefore not literature-verified
