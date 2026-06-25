# Ameliorating Inventory Instances

These JSON files describe benchmark/problem instances for the Pahr and Grunow ameliorating-inventory family. The primary numeric anchor is the companion-code perfect-information steady-state LP `max_reward`, an upper bound on long-run average profit, not an achievable policy optimum.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `pahr_grunow2025_spirits_0001.json` | `companion_code` | Checked-in dataset, Rust/Python verified |
| `pahr_grunow2025_spirits_0002.json` | `companion_code` | Checked-in dataset, Rust verified; Python binding pending |
| `pahr_grunow2025_spirits_1002.json` | `companion_code` | Checked-in dataset, Rust verified; Python binding pending |
| `pahr_grunow2025_port_wine.json` | `companion_code` | Checked-in dataset, Rust/Python verified |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test -p invman_rust --lib problems::ameliorating_inventory::tests::verification -- --nocapture
```

Caveat: LP values are upper bounds. Report policy results as gap-to-bound, never as beating the bound. Long expected-revenue and slope tables remain in dataset files or upstream companion code, not duplicated here.
