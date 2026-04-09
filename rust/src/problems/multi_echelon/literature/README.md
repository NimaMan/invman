# Literature

This folder documents the public benchmark family used for `multi_echelon`.

## Carried Literature References

- Gijsbrechts et al. (2022), Section 7.1 and Table 3
- Van Roy et al. (1997), two-echelon case study

## What Is Verified

- Setting parameters for the two Gijs benchmark settings
- Published policy rows carried from Gijs:
  - comparator family: constant base-stock over the Van Roy action grid
  - published learned-policy row: A3C
- Reported A3C savings over constant base-stock:
  - setting 1: `8.95% +/- 0.13%`
  - setting 2: `12.09% +/- 0.39%`
- Reported Van Roy savings: approximately `10%`
- Published Van Roy case-study benchmark row:
  - constant base-stock levels `(330, 23)`
  - average cost `1302`

## Repo Algorithm Status

- `constant_base_stock` in the carried Gijs settings: `literature_verified = false`
  - reason: Gijs publish only savings percentages, not absolute constant base-stock means, and the
    separate open Van Roy case-study heuristic row is not yet numerically reproduced by the repo
- repo exact verifier: `literature_verified = false`
  - reason: it is a repo-native reduced verifier, not a published exact benchmark row
- published A3C and Van Roy NDP rows are carried as published policy rows, not as verified repo
  algorithms

## What Is Not Public

- Gijs do not publish absolute mean costs for each setting.
- Therefore, the repo compares savings percentages against the reproduced constant base-stock benchmark rather than asserting exact literature costs for the two settings.
- Any absolute base-stock costs shown in repo experiment reports are reproduced repo numbers, not quoted Gijs numbers.
- The open Van Roy case-study row is carried for reference, but it is not treated as literature-verified because the current executable transcription does not reproduce the published cost.
