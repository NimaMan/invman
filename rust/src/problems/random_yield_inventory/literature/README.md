# Literature

Current literature anchors for `random_yield_inventory`:

- Yan et al. 2026
- Inderfurth and Kiesmuller 2015
- Chen et al. 2018

Repo interpretation:

- supply-side uncertainty is the primary missing axis carried by this family
- the first benchmark slice uses reduced discrete instances for exact verification

Use `references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature families

Current status: this package is not literature-verified.

The reasons are encoded directly in `literature/references.rs`:

- Yan 2026: exact model-family anchor, but no public reusable benchmark row recovered
- Chen 2018: policy anchor, but no public reusable benchmark row recovered
- Inderfurth 2015: public numbers exist, but only for related random-yield models rather than this
  repo's all-or-nothing executable formulation
