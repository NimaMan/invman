# Literature

Cited papers for `spare_parts_inventory` (all metadata independently verified
2026-05-31 against Crossref / publisher PDFs):

- Kranenburg (2006), Chapter 5 exact lateral-transshipment benchmark
  - LITERATURE-VERIFIED: the analytical module `kranenburg_lateral_transshipment.rs`
    re-derives Table 5.2 and the test reproduces every printed row within tolerance 0.02
  - this is a continuous-review, METRIC-style multi-location model and is STRUCTURALLY
    DIFFERENT from the trainable `env.rs`; the verification covers this analytical module
    only and says nothing about `env.rs`
- the spare-parts review carried in `references.rs` (no reusable numbers)
- Zhou et al. 2024 (motivation only, no carried numbers)
- van der Haar et al. 2025 (motivation only, no carried numbers)
- van Oers et al. 2024 table-only benchmark catalog
  - NOT literature-verified: recorded constants only, no executable env/solver re-runs
    them (frozen snapshot test). Kept as a catalog target for a future serial env.

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
