# Literature

Current literature anchors for `spare_parts_inventory`:

- Kranenburg (2006), Chapter 5 exact lateral-transshipment benchmark
  — LITERATURE-VERIFIED: `kranenburg_lateral_transshipment.rs` reproduces all 35
  published Table 5.2 rows (worst absolute deviation 0.005 vs the 0.02 table-rounding
  tolerance). This is the only block whose published numbers are recomputed by a repo
  solver rather than stored.
- the spare-parts review carried in `references.rs` — motivational only, no numbers
- Zhou et al. 2024 — motivational only, no numbers
- van der Haar et al. 2025 — motivational only, no numbers
- van Oers et al. 2024 — table-only benchmark catalog: Table 1 rows stored exactly as
  published; no repo solver re-derives them yet

Repo interpretation:

- repairable spares with installed-base failures, repair returns, and procurement lead times
- adjacent literature subfamilies may live here when the paper publishes benchmark numbers that
  can be carried and verified exactly

Use `references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes
- Kranenburg Table 5.2 reference rows
