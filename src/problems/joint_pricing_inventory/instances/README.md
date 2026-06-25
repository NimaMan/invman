# Joint Pricing Inventory Instances

This directory stores one JSON file per joint pricing inventory instance. The target executable model is the repo finite-horizon, zero-lead-time joint order-and-price environment with cost equal to procurement plus holding plus stockout minus revenue, with terminal salvage credit.

Current literature coverage has no strict published-number anchor for this exact env. Federgruen-Heching and Yin-Rajaram rows are important context, but they require backlog, fixed-cost, Markov-demand, or continuous-price extensions before executable comparison.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `qin2022_reduced_exact_verifier.json` | `faithful_unverified` | Exact DP self-consistency |
| `zhou2022_style_price_ladder.json` | `faithful_unverified` | Repo-native Poisson benchmark |
| `federgruen_heching_1999_dress_base_table.json` | `table_only` | Not executable in current env |
| `generated_exact_high_stockout_3p_T5.json` | `generated` | Generated exact finite-support row |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python scripts/joint_pricing_inventory/validate_against_exact_dp.py
```
