# Divergent Special-Delivery Instances

This directory is the machine-readable instance catalog for divergent special-delivery systems. It is scoped to this multi-echelon family and intentionally excludes the lost-sales problem family.

Generated instances are stress tests and must not be mixed into literature leaderboards. Cheng-style rows require a new event-order mode before they can move beyond metadata/table-only status.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `cheng2023_rbf_dqn_setting2_event_order.json` | `table_only` | Requires Cheng event-order dynamics |
| `cheng2023_rbf_dqn_setting3_event_order.json` | `table_only` | Requires Cheng event-order dynamics |
| `generated_gijs_setting1_low_wait_prob.json` | `generated` | Synthetic stress case |
| `generated_gijs_setting2_high_penalty.json` | `generated` | Synthetic stress case |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test --manifest-path /home/nima/code/ml/invman/Cargo.toml divergent_special_delivery -- --nocapture
```
