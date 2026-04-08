# Literature

Current literature anchor for `one_warehouse_multi_retailer`:

- Kaynov et al. 2024

Repo interpretation:

- divergent one-warehouse multi-retailer control
- allocation is part of the action structure, not a post-processing detail

Use `references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- the carried Table A.3 instances and benchmark-policy names

Current benchmark reproduction notes:

- `literature_verified = true` means the instance parameters and benchmark rows are carried from public literature
- the current script-side reproduction for Kaynov Table A.3 uses a mean-filled warm start and the fixed 100-period, 1000-trajectory protocol from the paper
- for symmetric three-retailer Kaynov instances, the strongest tree-policy results use a symmetric target interface rather than separate per-retailer order outputs
