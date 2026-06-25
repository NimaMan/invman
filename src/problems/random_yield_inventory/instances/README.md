# Random Yield Inventory Instances

This directory contains machine-readable instances for the finite-horizon, backlogged, all-or-nothing random-yield model implemented in `src/problems/random_yield_inventory`.

Current caveat: Yan et al. (2026) and Chen et al. (2018) match the all-or-nothing formulation but expose no public reusable numeric tables. Inderfurth and Kiesmueller (2015) is related random-yield/LIR literature, but it uses binomial or proportional yield and infinite-horizon average cost, so its numbers are not comparable to this environment.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `reduced_exact_verification_instance.json` | `faithful_unverified` | Exact DP self-consistency |
| `yan2026_style_lt2_p075_discounted.json` | `faithful_unverified` | Repo-native policy benchmark |
| `generated_exact_lt2_p050_low_yield_short_horizon.json` | `generated` | Generated exact-DP target |
| `generated_exact_lt2_p075_bimodal_demand.json` | `generated` | Generated exact-DP target |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python -c "import invman_rust as ir; s=ir.random_yield_inventory_exact_dp_summary(); assert abs(s['optimal_discounted_cost']-40.05989760985441)<1e-9"
```
