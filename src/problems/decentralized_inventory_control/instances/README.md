# Decentralized Inventory Control Instances

This directory is the machine-readable instance catalog for Beer-Game-style decentralized serial inventory control. It is an instance catalog, not a benchmark-result report.

Caveats:

- The Sterman/Edali-Yasarcan `204` verifies only the closed-form board-game port, not `env.rs`.
- Current `env.rs` gives `378` for Sterman anchor-and-adjust and best simple base-stock `278` on the classic 36-week path.
- Oroojlooyjadid companion-code rows use a related Beer Game implementation with different timing and action conventions.
- Mousa et al. rows include price, replenishment cost, capacity, and profit terms not represented in current `env.rs`.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `edali2014_sterman_classic_36w_closedform.json` | `companion_code` | Closed-form board-game port |
| `repo_exact_2agent_bernoulli_t3.json` | `generated` | Repo-native exact DP fixture |
| `oroojlooy2022_c4_8_lit_case.json` | `companion_code` | Companion-code structure, model mismatch |
| `mousa2024_serial4_table_b4.json` | `table_only` | Requires model extensions |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python scripts/decentralized_inventory_control/measure_env_vs_closedform.py
cargo test decentralized_inventory_control -- --nocapture
```
