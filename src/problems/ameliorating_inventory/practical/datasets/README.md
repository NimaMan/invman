# ameliorating_inventory practical datasets

Checked-in perfect-information LP datasets mirroring the Pahr & Grunow (2025) companion
repository (https://github.com/amelioratinginventory/ameliorating_inventory) per-instance
`config.json` / `expected_revenue.json` / `upper_bound.json`.

Each file is a whitespace-delimited key/value text blob (see `lp_dataset_loader.rs` for the
format) carrying the instance parameters, the per-product expected-revenue and slope tables
aligned to the production grid `0..sales_bound step production_step_size`, and the published
`published_max_reward` anchor that `tests/verification.rs` reproduces.

Files:

- `spirits_0001_perfect_information_lp.txt` — companion default spirits instance (10 ages,
  3 products, target ages [2,4,6], capacity 50, holding 2.5, no blending).
  Published upper bound `max_reward = 1991.9344293376805`.
- `port_wine_perfect_information_lp.txt` — port-wine industry case study (25 ages, 2 products,
  target ages [9,19], blending enabled). Published upper bound `max_reward = 2444.8010643781136`.

Both bounds are reproduced (gap < 1e-7) by re-solving the in-crate `microlp` simplex in
`tests/verification.rs`.
