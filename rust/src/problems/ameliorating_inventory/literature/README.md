# Literature

Current literature anchors for `ameliorating_inventory`:

- Pahr and Grunow 2025
- the companion public repository carried in `literature/references.rs`

Current status:

- not literature-verified

Why:

- the paper and the public repository define a richer executable model than the current Rust
  package
- the companion defaults use ten age classes and three products, while the current Rust primary
  instance uses five age classes and two products
- the paper environment also includes stochastic sales-price and decay processes that are fixed in
  the current Rust approximation

Use `literature/references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- benchmark-policy names
- literature notes that explain the formulation gap
