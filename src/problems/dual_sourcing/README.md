# Dual Sourcing

This package implements the small-scale dual-sourcing benchmark family used by Gijsbrechts et al.
(2022).

## Formulation

- one regular supplier with lead time `l_r`
- one expedited supplier with lead time `l_e = 0`
- linear sourcing costs `c_r < c_e`
- end-of-period holding cost `h`
- end-of-period backlog cost `b`
- demand uniform on `{0, 1, 2, 3, 4}`

The package uses the six small benchmark settings carried by Gijs:

- `l_r in {2, 3, 4}`
- `c_e in {105, 110}`
- shared parameters `c_r = 100`, `h = 5`, `b = 495`

## Literature

Primary references:

- Gijsbrechts et al. (2022), M&SOM 24(3):1349-1368
- Veeraraghavan and Scheller-Wolf (2008), Operations Research 56(4):850-864
- Sheopuri et al. (2010), Operations Research 58(3):734-745

Executable literature references carried here:

- `dual_l2_ce105`
- `dual_l2_ce110`
- `dual_l3_ce105`
- `dual_l3_ce110`
- `dual_l4_ce105`
- `dual_l4_ce110`

Published benchmark comparators for this family are:

- `optimal_dp`
- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`
- `lp_adp`
- `a3c`

The public Gijs row-level numbers we carry for these settings are the Figure 9 optimality-gap
labels, not a full absolute-cost table.

## Heuristics

Implemented heuristic families:

- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`

Published benchmark takeaways:

- Gijs report that A3C is within `2%` of optimal on all six settings.
- Gijs show `capped_dual_index` as the strongest heuristic benchmark on this test family.
- Veeraraghavan and Scheller-Wolf are the source of the small-scale benchmark family.
- Sheopuri et al. extend the classical dual-sourcing policy family beyond the original dual-index
  rule.

## Verification

The package also carries a bounded dynamic-programming verifier for the six small settings.

This verifier is not a proof-level exact solver outside the chosen truncation box, so benchmark
claims should treat it as a bounded finite-state reference, not as a universal exact-optimality
claim for the unbounded model.

The repo already carries the six published Figure 9 optimality-gap labels in
`literature/references.rs`. Tests in `verification/tests.rs` freeze the full six-row table and
execute the canonical `dual_l2_ce105` benchmark instance against those labels.

## Policy Interface

The environment interface remains raw-state first:

- env/state builders expose raw reduced-state values
- normalization and feature construction happen in rollout or policy code, not inside the
  environment

## Benchmark Protocol

For this family, the literature-aligned comparison is:

- report repo heuristic or learned-policy costs against the bounded DP reference
- compare reproduced optimality gaps to the published Figure 9 gap labels

## Package Layout

- `experiments/`: paper-facing benchmark notes and future checked-in report snapshots
- `literature/`: literature anchors and the carried Figure 9 gap table
- `practical/`: practical benchmark placeholders when a canonical practical slice exists
- `verification/`: verification scope and the executable check strategy
