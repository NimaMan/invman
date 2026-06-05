# Practical

This folder is the practical-benchmark home for `ameliorating_inventory`.

Current state:

- `datasets/` carries the checked-in perfect-information LP datasets mirroring the Pahr & Grunow
  (2025) companion repository (`spirits_0001`, `port_wine`): instance parameters, per-product
  expected-revenue / slope tables, and the published `max_reward` upper-bound anchor.
- those datasets back the executing literature-verification test in `tests/verification.rs`, which
  re-solves the perfect-information LP and reproduces the published bounds (gap < 1e-7).
- see `datasets/README.md` for the file inventory and format.

Still to add when available:

- benchmark notes for the practical slice
- checked-in report snapshots when a canonical practical benchmark exists
