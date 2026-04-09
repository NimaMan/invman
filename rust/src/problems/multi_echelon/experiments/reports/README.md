# multi_echelon Paper Benchmark Report

- source: Gijsbrechts et al. (2022), Manufacturing & Service Operations Management 24(3):1349-1368
- url: https://doi.org/10.1287/msom.2021.1064
- instances: `2`
- policy family: depth `1` `axis_aligned` soft tree with `linear` leaves
- literature evaluation: `100` sample paths of `100000` periods each
- literature baseline: constant base-stock with `min_shortage` allocation
- baseline note: Repo comparator is the best constant base-stock policy searched over the carried Van Roy action grid. The Gijs text clearly states the learned policy uses that grid, but the constant-base-stock search domain in the paper still needs final clarification.
- training budget: `20` CMA-ES episodes of length `5000`

## Reporting Rule

- `literature_verified` applies only to repo exact or heuristic algorithms.
- Published A3C / PPO / NDP rows from papers are carried as published rows, not as verified repo algorithms.
- Repo reproduced absolute costs are shown separately from published literature numbers.

## Aggregate

- beats published A3C savings on `0` / `2` settings
- beats published Van Roy savings on `0` / `2` settings
- mean soft-tree savings vs repo constant base-stock: `0.004%`
- mean gap vs published A3C savings: `-10.516` percentage points

## Repo Algorithm Verification

| Repo Algorithm | literature_verified | Verification Anchor | Note |
| --- | --- | --- | --- |
| `constant_base_stock` | `False` | `none` | The paper reports only relative savings for the two Gijs settings, not absolute constant base-stock means. The open Van Roy case-study heuristic row is carried separately, but the current executable transcription does not yet reproduce that published cost. |

## Published Numbers Confirmed

| Instance | Published Constant Base-Stock Cost | Published A3C Savings | Published Van Roy Savings |
| --- | ---: | ---: | ---: |
| `gijsbrechts2022_setting1` | `not reported` | `8.95% +/- 0.13%` | `~10.00%` |
| `gijsbrechts2022_setting2` | `not reported` | `12.09% +/- 0.39%` | `~10.00%` |

## Per Instance

Repo reproduction benchmark:

| Instance | Base-Stock Cost | Soft Tree Cost | Soft Tree Savings | 95% Half-Width | Published A3C Savings | Published Van Roy Savings | Gap vs A3C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gijsbrechts2022_setting1` | `3087.965` | `3087.880` | `0.003%` | `0.002%` | `8.95% +/- 0.13%` | `~10.00%` | `-8.947` |
| `gijsbrechts2022_setting2` | `3793.797` | `3793.613` | `0.005%` | `0.002%` | `12.09% +/- 0.39%` | `~10.00%` | `-12.085` |
