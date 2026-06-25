# Dual Sourcing Instances

This directory is the machine-readable instance catalog for the backorder dual-sourcing problem. It is an instance catalog, not a benchmark-result report.

Classification meanings:

- `strict_literature`: an in-repo executable check reproduces a peer-reviewed printed quantity.
- `companion_code`: exact parameters come from an open companion implementation, but this repo does not yet re-run the published value.
- `table_only`: a printed or companion table value is stored, but no in-repo re-run currently reproduces it.
- `faithful_unverified`: the instance matches a published model family, but no external numeric anchor is stored.
- `generated`: repo-native stress case with no external benchmark value.

The current Gijsbrechts et al. rows use `cr=100` and carry published optimality gaps only. The Böttcher et al. rows use `cr=0` and printed absolute average costs, so they must not be mixed with the Gijs rows without an explicit cost-shift convention.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `bottcher2023_lr2_ce10_b95_u08.json` | `companion_code` | Needs bounded-DP loader against printed cost 37.24 |
| `bottcher2023_lr3_ce10_b95_u04.json` | `companion_code` | Needs bounded-DP loader against printed cost 20.34 |
| `generated_dual_l2_ce110_b50_u08_catC.json` | `generated` | Repo taxonomy probe; no literature value |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python scripts/dual_sourcing/validate_reference_grid.py --references dual_l2_ce105 dual_l2_ce110 --with_optimal_dp
```
