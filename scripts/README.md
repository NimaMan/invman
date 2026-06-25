# Scripts

Runnable scripts are grouped by problem family or shared workflow. Do not add
loose `.py` or `.sh` files directly under this directory.

- `benchmark_baselines/` validates and reports literature baseline registries.
- `experiments/` contains generic experiment entry points.
- `rust/` contains Rust/PyO3 build helpers.
- `seed_robust/` contains cross-problem seed-robust audit queues.
- Problem folders such as `lost_sales/`, `dual_sourcing/`, and
  `one_warehouse_multi_retailer/` own their respective benchmark and
  autoresearch runners.

