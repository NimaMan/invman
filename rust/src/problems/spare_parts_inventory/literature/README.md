# Literature

Cited papers for `spare_parts_inventory` (all metadata independently verified
2026-05-31 against Crossref / publisher PDFs):

- **Kranenburg, A. A. (2006)**, *Spare parts inventory control under system
  availability constraints*, PhD thesis, Technische Universiteit Eindhoven
  (DOI 10.6100/IR616052; open-access PDF at
  `pure.tue.nl/ws/files/2461454/200612097.pdf`). Chapter 5 "Lateral transshipment:
  An exact analysis", Tables 5.1-5.3. LITERATURE-VERIFIED benchmark family.
- **van Oers, J., Tanil, I. & Basten, R. (2024)**, *Numerical Analysis of A Spare
  Parts Supply Chain With Additive Manufacturing*, IFAC-PapersOnLine 58(19),
  1006-1011 (DOI 10.1016/j.ifacol.2024.09.144). Table-only benchmark catalog.
- **Zhang, S., Huang, K. & Yuan, Y. (2021)**, *Spare Parts Inventory Management:
  A Literature Review*, Sustainability 13(5), article 2460
  (DOI 10.3390/su13052460). Motivational only, no numbers.
- **Zhou, Y., Guo, K., Yu, C. & Zhang, Z. (2024)**, *Optimization of multi-echelon
  spare parts inventory systems using multi-agent deep reinforcement learning*,
  Applied Mathematical Modelling 125, 827-844 (DOI 10.1016/j.apm.2023.10.039).
  Motivational only, no numbers.
- **van der Haar, J. F., van Jaarsveld, W., Basten, R. J. I. & Boute, R. N.**,
  *Industrializing Deep Reinforcement Learning for Operational Spare Parts
  Inventory Management*, SSRN working paper 4999374 (posted 2024; the const name
  `VAN_DER_HAAR_2025_REFERENCE` uses an approximate 2025 vintage). Motivational
  only, no numbers.

## Verification status (honest, audited 2026-05-31)

- **Kranenburg (2006) Table 5.2 — LITERATURE-VERIFIED.** This is the only block
  whose published numbers are recomputed by a repo solver, not just stored. The
  from-scratch analytical solver `kranenburg_lateral_transshipment.rs` reproduces
  all 35 published Table 5.2 rows (Situation 1 separate stock points vs Situation 3
  lateral transshipment: optimal randomized stock `R*`, cost `C(R*)`, and the cost
  ratio), worst absolute deviation 0.005 against the 0.02 table-rounding tolerance.
  The base case anchors `R1*=9.09, C1=91.90, R3*=6.10, C3=63.00, ratio=1.46`. ALL
  35 stored anchor rows in `references.rs` were confirmed verbatim against the
  thesis PDF during this audit, and Table 5.1 base-case parameters match exactly.
- **van Oers et al. (2024) Table 1 — TABLE-ONLY (recorded-as-published, NOT
  reproduced).** The Table 1 rows are stored exactly as transcribed; no repo solver
  re-derives them. CAVEAT: the individual Table 1 cell values (costs, readiness,
  base-stock levels) were NOT independently re-confirmed against the published
  table during the audit (full text paywalled) — only the bibliographic metadata
  (authors / venue / volume / issue / pages / DOI) was verified. Note also that the
  `LiteratureBenchmarkScenario` structs set `literature_verified: true` with
  `verification_source = "published_benchmark_table_from_literature"`; that boolean
  flag means only "transcribed from a published table", NOT "reproduced by a
  solver". Read it as table-only.
- **Single-echelon repairable MDP (primary + verification instances) — NOT
  literature-verified.** Repo-native; flagged in code as
  `repo_exact_solver_not_verified_against_literature`. It is an internal
  self-consistency anchor only (the exact DP must weakly dominate both carried
  heuristics). The MDPI review, Zhou (2024), and van der Haar (2024) are
  motivational framing for this family and carry no benchmark numbers.

Repo interpretation:

- repairable spares with installed-base failures, repair returns, and procurement
  lead times
- adjacent literature subfamilies may live here when the paper publishes benchmark
  numbers that can be carried (and, where a solver exists, verified) exactly

Use `references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes
- Kranenburg Table 5.2 reference rows
