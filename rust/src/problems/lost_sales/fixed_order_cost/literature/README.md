# Literature

Current literature anchor for `lost_sales_fixed_order_cost`:

- Bijvank, Bhulai, and Huh (2015), Table 1

Executable literature verification currently uses:

- the published Poisson validation instance `bijvank2015_table1_l2_p14_k5`
- the published optimal average cost
- the published best `(s,S)`, `(s,nQ)`, and modified `(s,S,q)` rows

Use `references.rs` as the source of truth for:

- literature metadata
- the carried Table 1 validation instance
- published benchmark-policy names and reported numbers

Current status:

- this package is literature-verified on the published Table 1 validation instance
- larger fixed-cost benchmark grids are not yet literature-verified
