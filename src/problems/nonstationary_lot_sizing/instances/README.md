# Nonstationary Lot-Sizing Instances

This directory contains one JSON file per nonstationary lot-sizing instance. The JSON files are the source of truth for instance parameters, provenance, carried reference values, and verification status.

This pass adds non-lost-sales backorder rows from the Dehaybe et al. companion testbed. The existing lost-sales rows remain in `references.rs` and are not the target of this refresh.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `dehaybe2024_backorder_lt2_b25_k320_constant_5.json` | `companion_code` | Simple `(s,S)` smoke-verified; rolling-DP blocked |
| `dehaybe2024_backorder_lt2_b25_k320_constant_10.json` | `companion_code` | Simple `(s,S)` smoke-verified; rolling-DP blocked |
| `dehaybe2024_backorder_lt2_b25_k320_seasonal_2.json` | `companion_code` | Simple `(s,S)` smoke-verified; rolling-DP blocked |
| `dehaybe2024_backorder_lt2_b25_k320_growth.json` | `companion_code` | Simple `(s,S)` smoke-verified; rolling-DP blocked |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test -p invman_rust nonstationary_lot_sizing
```

Caveat: current Rust rolling-DP verification supports Poisson demand only. These backorder CV-normal rolling-DP companion values are stored as reference values but are not locally verified until that solver is extended or the original Julia verifier is run.
