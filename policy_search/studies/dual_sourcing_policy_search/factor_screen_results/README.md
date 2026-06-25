# Dual-Sourcing Factor Screen

Run tag: `dual_sourcing_factor_screen_v1`
Budget: `screening`

This note uses the six Gijs Figure 9 benchmark rows as a controlled policy-design testbed.
The goal is not to defend one fixed policy class, but to identify which design choices drive learned-policy performance.

## Aggregate Ranking

| policy | mean gap vs best heuristic (%) | wins vs best heuristic | control family | structure |
| --- | ---: | ---: | --- | --- |
| `tree_axis_constant_smallcap_delta` | `0.7529` | `1/6` | `smallcap_delta` | `tree_axis_constant` |
| `tree_smallcap_delta` | `1.1033` | `0/6` | `smallcap_delta` | `tree_oblique_linear` |
| `tree_capped_delta` | `1.7565` | `0/6` | `capped_delta` | `tree_oblique_linear` |
| `linear_smallcap_delta` | `2.3757` | `0/6` | `smallcap_delta` | `linear` |
| `tree_capped_dual_index` | `2.7118` | `0/6` | `capped_dual_index` | `tree_oblique_linear` |

## Best Policy By Benchmark Row

| reference | best policy | gap vs best heuristic (%) |
| --- | --- | ---: |
| `dual_l2_ce105` | `tree_axis_constant_smallcap_delta` | `0.0518` |
| `dual_l2_ce110` | `tree_capped_delta` | `0.7020` |
| `dual_l3_ce105` | `tree_axis_constant_smallcap_delta` | `0.0259` |
| `dual_l3_ce110` | `tree_axis_constant_smallcap_delta` | `0.2090` |
| `dual_l4_ce105` | `tree_axis_constant_smallcap_delta` | `-0.1052` |
| `dual_l4_ce110` | `tree_axis_constant_smallcap_delta` | `0.3564` |

## Factor Effects

### Factorizing regular targets (`s_r = s_e + delta_r`) versus unfactorized capped dual-index targets

Average improvement in gap when moving from `tree_capped_dual_index` to `tree_capped_delta`: `0.9553` percentage points.

| reference | left gap (%) | right gap (%) | improvement (pp) |
| --- | ---: | ---: | ---: |
| `dual_l2_ce105` | `3.7859` | `1.8577` | `1.9282` |
| `dual_l2_ce110` | `3.8660` | `0.7020` | `3.1640` |
| `dual_l3_ce105` | `1.3280` | `1.7778` | `-0.4498` |
| `dual_l3_ce110` | `2.8907` | `2.8390` | `0.0518` |
| `dual_l4_ce105` | `1.3292` | `1.3292` | `0.0000` |
| `dual_l4_ce110` | `3.0709` | `2.0333` | `1.0377` |

### Adding a small discrete regular-cap grid on top of the capped-delta tree

Average improvement in gap when moving from `tree_capped_delta` to `tree_smallcap_delta`: `0.6532` percentage points.

| reference | left gap (%) | right gap (%) | improvement (pp) |
| --- | ---: | ---: | ---: |
| `dual_l2_ce105` | `1.8577` | `2.2385` | `-0.3808` |
| `dual_l2_ce110` | `0.7020` | `1.7373` | `-1.0353` |
| `dual_l3_ce105` | `1.7778` | `0.4001` | `1.3778` |
| `dual_l3_ce110` | `2.8390` | `1.5927` | `1.2463` |
| `dual_l4_ce105` | `1.3292` | `0.2768` | `1.0525` |
| `dual_l4_ce110` | `2.0333` | `0.3742` | `1.6591` |

### Replacing the oblique linear tree with an axis-aligned constant-leaf tree on the same small-cap control family

Average improvement in gap when moving from `tree_smallcap_delta` to `tree_axis_constant_smallcap_delta`: `0.3504` percentage points.

| reference | left gap (%) | right gap (%) | improvement (pp) |
| --- | ---: | ---: | ---: |
| `dual_l2_ce105` | `2.2385` | `0.0518` | `2.1868` |
| `dual_l2_ce110` | `1.7373` | `3.9795` | `-2.2422` |
| `dual_l3_ce105` | `0.4001` | `0.0259` | `0.3742` |
| `dual_l3_ce110` | `1.5927` | `0.2090` | `1.3837` |
| `dual_l4_ce105` | `0.2768` | `-0.1052` | `0.3820` |
| `dual_l4_ce110` | `0.3742` | `0.3564` | `0.0178` |

## Current Reading

- Hardest row under the current search surface: `dual_l2_ce110` with best learned gap `0.7020%`.
- Easiest row under the current search surface: `dual_l4_ce105` with best learned gap `-0.1052%`.
- The dominant factor is control geometry, not just parameter count.
- Factorized dual-index controls help more than staying in unfactorized target coordinates.
- A small discrete cap on regular orders is not enough by itself; it helps most when paired with a tighter policy geometry.
- On the hard rows, tighter tree geometry can beat a more flexible oblique tree on the same control family.
- The right next step is to keep the good control family and search more policy classes on top of it, not to retreat to raw direct-order outputs.

## Promotion Candidates

- Promote `tree_axis_constant_smallcap_delta` as the default six-row search family. Its mean gap is `0.7529%` and it is the best policy on five of the six rows.
- Keep `tree_capped_delta` alive as the main alternate family. It wins `dual_l2_ce110` and is the best non-axis-constant option on the easy rows.
- Deprioritize `tree_capped_dual_index`: it is consistently worse than the factorized variants.
- Deprioritize `linear_smallcap_delta` as a main family: the control family is reasonable, but the backbone is too weak to compete consistently.

## Follow-Up Axis-Linear Probes

The factor screen suggested trying axis-aligned linear leaves on the factorized delta control families.
Those probes show that linear leaves are useful, but not as a universal replacement.

| reference | best factor-screen family | factor-screen gap (%) | best axis-linear follow-up | follow-up gap (%) | delta (pp) |
| --- | --- | ---: | --- | ---: | ---: |
| `dual_l2_ce105` | `tree_axis_constant_smallcap_delta` | `0.0518` | `tree_axis_linear_smallcap_delta` | `-0.0621` | `0.1139` |
| `dual_l2_ce110` | `tree_capped_delta` | `0.7020` | `tree_axis_linear_capped_delta` | `-0.0831` | `0.7851` |
| `dual_l3_ce105` | `tree_axis_constant_smallcap_delta` | `0.0259` | `tree_axis_linear_capped_delta` | `1.7778` | `-1.7520` |
| `dual_l3_ce110` | `tree_axis_constant_smallcap_delta` | `0.2090` | `tree_axis_linear_smallcap_delta` | `1.0613` | `-0.8523` |
| `dual_l4_ce105` | `tree_axis_constant_smallcap_delta` | `-0.1052` | `tree_axis_linear_smallcap_delta` | `0.9317` | `-1.0369` |
| `dual_l4_ce110` | `tree_axis_constant_smallcap_delta` | `0.3564` | `tree_axis_linear_smallcap_delta` | `0.8692` | `-0.5128` |

### Updated Direction

- Axis-linear follow-ups improve on the factor-screen best family for `dual_l2_ce105`, `dual_l2_ce110`.
- Axis-linear follow-ups are worse than the factor-screen best family for `dual_l3_ce105`, `dual_l3_ce110`, `dual_l4_ce105`, `dual_l4_ce110`.
- The common ingredient across the winning rows is still the factorized capped-delta control surface.
- The split is in policy geometry: `l_r = 2` rows benefit from axis-aligned linear leaves, while `l_r in {3,4}` still prefer the tighter `tree_axis_constant_smallcap_delta` family.
- The strongest next design is a lead-time-conditioned portfolio or mixture on top of the same factorized control basis, rather than one universal backbone.
