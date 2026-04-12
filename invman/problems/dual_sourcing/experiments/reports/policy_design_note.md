# Dual-Sourcing Policy Design Note

This note records the current `l_r=3` policy-design finding for the Gijs Figure 9 dual-sourcing family.

## Main point

Performance in dual sourcing is highly sensitive to the policy parameterization, even when the policies
share the same state and training budget.

The strongest current result is not the most flexible tree. It is a tightly structured, low-parameter
policy:

- axis-aligned soft tree
- constant leaves
- capped-dual-index-delta small-cap control grid

This matters because it supports the paper claim that compact learned policies can match or nearly match
the best benchmark heuristics, and sometimes beat them on the repo evaluation protocol.

## Candidate policies

The relevant structured dual-sourcing policies now include:

- `soft_tree_capped_dual_index_delta_smallcap_targets`
  - oblique splits
  - linear leaves
  - control family `(s_e, delta_r, cap_r)` with `cap_r in {1,2,3,4,6,8,12}`
- `soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets`
  - axis-aligned splits
  - constant leaves
  - same small-cap control family

The second policy is materially smaller and empirically stronger on the hard `l_r=3` rows.

## Parameter counts at `l_r=3`

For the `l_r=3` reduced state, the input dimension is `3` and the control dimension is also `3`.

- depth-1 axis-aligned constant tree: `10` parameters
- depth-2 axis-aligned constant tree: `24` parameters
- depth-3 axis-aligned constant tree: `52` parameters
- depth-2 oblique linear tree: `60` parameters

These counts come directly from the soft-tree parameterization:

- internal-node split weights and biases
- leaf-local control outputs

## Current `l_r=3` screening results

Rows below report relative gap to the best heuristic on the same evaluation protocol.

| policy | params | `dual_l3_ce105` | `dual_l3_ce110` |
| --- | ---: | ---: | ---: |
| oblique linear small-cap tree, depth 2 | 60 | `0.4001%` | `1.5927%` |
| axis-aligned constant small-cap tree, depth 2, `t=0.25` | 24 | `0.0259%` | `0.2090%` |
| axis-aligned constant small-cap tree, depth 1, `t=0.40` | 10 | `-0.1276%` | pending targeted search |
| axis-aligned constant small-cap tree, depth 2, `t=0.15` | 24 | `-0.1049%` | pending targeted search |

Interpretation:

- the wider oblique linear tree is not the best design here
- a simpler heuristic-like control surface is easier for CMA-ES to search
- on `dual_l3_ce105`, very small trees already beat the best heuristic on the repo protocol
- on `dual_l3_ce110`, the best completed result so far is the 24-parameter axis-aligned constant tree

## Paper-facing claim

A defensible paper claim from the current evidence is:

"In dual sourcing, policy design matters at least as much as raw parameter count. Tightly structured
learned policies with as few as `10` to `24` parameters can match or outperform the best benchmark
heuristics on some literature instances, while larger and more flexible trees can perform materially
worse."

## Result files

- `dual_l3_ce105_axis_constant_probe/screening_summary.json`
- `dual_l3_ce110_smallcap_shape_sweep/screening_summary.json`
- `dual_l3_axis_constant_temp_sweep/screening_summary.json`
