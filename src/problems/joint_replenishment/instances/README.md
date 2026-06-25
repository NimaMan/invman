# Joint Replenishment Instances

This directory contains one JSON file per joint-replenishment problem instance.

The current instance set is the small-scale Vanvuchelen, Gijsbrechts, and Boute (2020) family: two items, zero lead time, backorders, full-truck capacity `V=6`, major cost `K=75`, discount factor `0.99`, and inclusive discrete-uniform demands. These rows are `table_only`: the paper gives the parameters, and setting 5 gives an action anchor, but it does not print absolute per-setting optimal costs.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `vanvuchelen2020_small_scale_setting_1.json` | `table_only` | Schema/table row only |
| `vanvuchelen2020_small_scale_setting_5.json` | `table_only` | Published action anchor at state `[5,0]` |
| `vanvuchelen2020_small_scale_setting_16.json` | `table_only` | Schema/table row only |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test joint_replenishment::verification::tests::value_iteration_reproduces_figure3_optimal_action --quiet
python -c "import invman_rust as ir; print(ir.joint_replenishment_exact_dp_summary())"
```
