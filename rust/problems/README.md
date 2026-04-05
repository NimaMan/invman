# Rust Problem Homes

This directory is the canonical non-code home for Rust-first problem families.

Use it together with `rust/src/problems/<problem>/`:

- `rust/src/problems/<problem>/` holds executable code only
- `rust/problems/<problem>/` holds literature notes, practical benchmark assets, and
  experiment definitions plus human-readable verification targets

Standard layout:

```text
rust/problems/<problem>/
  README.md
  literature/
  practical/
    datasets/
    reports/
  experiments/
  verification/
```

The first families migrated to this structure are:

- `perishable_inventory`
- `nonstationary_lot_sizing`

Other Rust-first families can adopt the same structure incrementally without changing their crate
code layout.

The default paper-facing file for a mature family is:

- `experiments/paper_benchmark.md`

That file defines the reported instances, CMA-ES-optimized policy families, heuristic comparators,
and whether an exact optimal benchmark exists for the reported slice.
