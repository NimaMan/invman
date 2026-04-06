# Rust Problems

This directory is the canonical home for Rust-first problem families.

Each family lives entirely under `rust/src/problems/<problem>/`, combining:

- executable code
- literature notes
- practical benchmark assets
- experiment definitions
- human-readable verification targets

Markdown convention:

- each folder uses a single markdown entrypoint
- that file is always `README.md`

Standard layout:

```text
rust/src/problems/<problem>/
  README.md
  literature/
  practical/
    datasets/
    reports/
  experiments/
    reports/
  verification/
  mod.rs
  env.rs
  heuristics/
  rollout.rs
  references.rs
  bindings.rs
  tests/
```

The first families migrated to this structure are:

- `perishable_inventory`
- `nonstationary_lot_sizing`

Other Rust-first families should migrate to the same structure so the code and artifact base stay
co-located.

The default paper-facing file for a mature family is:

- `experiments/README.md`

That file defines the reported instances, CMA-ES-optimized policy families, heuristic comparators,
and whether an exact optimal benchmark exists for the reported slice.
