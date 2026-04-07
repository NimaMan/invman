# Literature

Current literature anchors for `spare_parts_inventory`:

- Kranenburg (2006), Chapter 5 exact lateral-transshipment benchmark
- the spare-parts review carried in `references.rs`
- Zhou et al. 2024
- van der Haar et al. 2025
- van Oers et al. 2024 table-only benchmark catalog

Repo interpretation:

- repairable spares with installed-base failures, repair returns, and procurement lead times
- adjacent literature subfamilies may live here when the paper publishes benchmark numbers that
  can be carried and verified exactly

Use `references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes
- Kranenburg Table 5.2 reference rows
