# Rust Problem Homes

This directory is the canonical non-code home for Rust-first problem families.

Use it together with `rust/src/problems/<problem>/`:

- `rust/src/problems/<problem>/` holds executable code only
- `rust/problems/<problem>/` holds literature notes, practical benchmark assets, and
  human-readable verification targets

Standard layout:

```text
rust/problems/<problem>/
  README.md
  literature/
  practical/
    datasets/
    reports/
  verification/
```

The first families migrated to this structure are:

- `perishable_inventory`
- `nonstationary_lot_sizing`

Other Rust-first families can adopt the same structure incrementally without changing their crate
code layout.
