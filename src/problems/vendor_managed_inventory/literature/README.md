# Literature

Primary anchor (corrected attribution, 2026-06-04):

- Sui, Z., A. Gosavi, and L. Lin (2010), *A Reinforcement Learning Approach for Inventory
  Replenishment in Vendor-Managed Inventory Systems With Consignment Inventory*, Engineering
  Management Journal 22(4): 44-53. DOI: 10.1080/10429247.2010.11431878.

Attribution correction:

- earlier revisions mis-attributed this DOI/title to "Giannoccaro and Pontrandolfo (2010)". That was
  wrong. The DOI and title belong to Sui/Gosavi/Lin (2010). All `references.rs` symbols are now
  `SUI_GOSAVI_LIN_2010_*`.

Public companion material (NOT the peer-reviewed paper):

- Gosavi instructor teaching case study (`vmi_case_study.pdf`): "CASE STUDY FOR VENDOR-MANAGED
  INVENTORY (BASED ON SUI, GOSAVI, & LIN, 2010)", Missouri Univ. of Science and Technology,
  Sept 7, 2010. It self-describes as class material and as "based on the journal article: Sui,
  Gosavi, and Lin (2010)".
- Gosavi MATLAB code for the same case (`vmi_newsvendor.m`).

The Rust identifiers `GIANNOCCARO_2010_REFERENCE`, `GIANNOCCARO_2010_NEWSVENDOR_WORKED_CASE`,
`GIANNOCCARO_2010_CASE_DEFINITIONS`, and `build_giannoccaro_2010_case` still carry the wrong name. They
were left unrenamed in this pass because they are referenced from `bindings.rs`, `rollout.rs`, and
`verification/tests.rs`, and renaming them requires a Rust rebuild + binding regeneration (out of
scope for a string-only citation fix). Renaming them to `SUI_GOSAVI_LIN_2010_*` is a tracked blocker.

- the repo-constructed 10-retailer/2-product truck-dispatch case definitions (a structural
  interpretation of the paper, not a transcribed published table)
- the Gosavi instructor-case single-retailer newsvendor worked example
- the repo-native reduced single-retailer primary instance and exact verifier instance

What is and is NOT literature-verified (per docs/rust.md):

- NOT literature-verified: no number printed in the peer-reviewed Sui/Gosavi/Lin (2010) paper is
  re-run by this family. The paper's experimental results table (pp. 44-53) is paywalled and not
  openly reproducible. `references.rs` carries `literature_verified = false` on both
  `SUI_GOSAVI_LIN_2010_REFERENCE` and `SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE`.
- reproduced exactly (but NOT counted as literature verification): the Gosavi INSTRUCTOR teaching
  case study worked example. Per the repo rule, an instructor/handout number is explicitly not
  literature verification. The reproduced order-up-to values (15 / 31.53 / 26.96) are Gosavi's own
  worked computation in the handout, not a results row printed in the peer-reviewed paper.

Why the full case table is dropped from the benchmark layer:

- the published profit rows are not openly accessible
- the demand-signal process is not defined precisely enough in any open source to reproduce the rows
- an earlier audit could not match the reproduced case-1 newsvendor profit to a published figure
  closely enough to anchor verification, and that figure was itself figure-only, not an openly
  available results table
