# Dual Sourcing

This package implements the small-scale dual-sourcing benchmark family used by Gijsbrechts et al. (2022) for the Veeraraghavan-Scheller-Wolf settings:

## Literature guidance

### Primary references

- Joren Gijsbrechts, Robert N. Boute, Jan A. Van Mieghem, and Dennis J. Zhang,
  *Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual
  Sourcing, and Multi-Echelon Problems*, Manufacturing & Service Operations Management, 2022.
- DOI: <https://doi.org/10.1287/msom.2021.1064>
- Veeraraghavan and Scheller-Wolf (2008), the dual-sourcing benchmark family used by the paper:
  <https://ideas.repec.org/a/inm/oropre/v56y2008i4p850-864.html>

### Published problem family

The Gijsbrechts dual-sourcing benchmark family uses:

- regular supplier lead time `lr in {2, 3, 4}`
- expedited lead time `le = 0`
- discrete uniform demand on `{0, 1, 2, 3, 4}`
- holding cost `h = 5`
- backlog cost `b = 495`
- regular cost `c_r = 100`
- expedited cost `c_e in {105, 110}`

### Published neural architecture

The paper uses the same fixed A3C backbone here as in lost sales and multi-echelon:

- four fully connected layers `[150, 120, 80, 20]`
- ReLU after each layer
- value regularization `0.25`
- four parallel learners
- gradient clipping `40`

### Published action design

The dual-sourcing action is two-dimensional:

- regular-source order quantity
- expedited-source order quantity

Repo implication:

- dual sourcing should not inherit a single scalar lost-sales-style `Q`
- if bounded policies are used here, the regular and expedited bounds should be explicit and
  source-specific policy parameters

Implemented heuristic families:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

The package also includes a bounded dynamic-programming solver over the reduced `lr`-dimensional state representation for correctness checks on the small-scale instances.

## Benchmark references

The canonical benchmark source for this package is:

- Gijsbrechts, Boute, Van Mieghem, and Zhang (2022), Section 6.2 / Figure 9

The package-level reference metadata is stored in:

- `reference_instances.py`

and records:

- the six literature settings
- the Gijs Figure 9 experiment grid
- the benchmark policy families
- the published benchmark claims that exist in the paper

Published benchmark comparators:

- optimal DP
- single-index
- dual-index
- capped dual-index
- tailored base-surge
- LP-ADP
- A3C

Published benchmark claim:

- A3C is within `2%` of optimal on all six small-scale settings
- capped dual-index is the strongest heuristic benchmark in the Gijsbrechts experiments

The literature does not provide a clean per-instance table of absolute costs for all six settings. Figure 9
does, however, print per-instance optimality-gap labels for the heuristic families and A3C, and those
published gap values are stored in `reference_instances.py`. Exact absolute costs in this repo therefore
remain repo-native benchmark values on the published problem family.

### Published Figure 9 Gap Labels

These are the rounded optimality gaps printed on the bars in Gijsbrechts et al. (2022), Figure 9:

| instance | capped dual-index | dual-index | single-index | tailored base-surge | A3C |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dual_l2_ce105` | 0.00 | 0.11 | 0.56 | 0.06 | 0.52 |
| `dual_l2_ce110` | 0.03 | 0.18 | 1.03 | 0.99 | 0.80 |
| `dual_l3_ce105` | 0.00 | 0.27 | 0.98 | 0.01 | 0.82 |
| `dual_l3_ce110` | 0.06 | 0.36 | 2.11 | 0.71 | 0.51 |
| `dual_l4_ce105` | 0.00 | 0.36 | 1.43 | 0.00 | 1.85 |
| `dual_l4_ce110` | 0.11 | 0.49 | 2.44 | 0.58 | 1.33 |

These gap labels are the main literature-grounded validation target for this package. The paper still
does not publish the corresponding absolute optimal or heuristic cost table.

If we require exact absolute heuristic costs as literature values, those still need a source beyond
Gijsbrechts Figure 9. Until such a table is located and verified, absolute costs produced by this repo
must remain labeled as repo-native.

### Validation Workflow

Use `scripts/dual_sourcing/validate_reference_grid.py` for the literature-facing check. It now reports:

- `bounded_dp.average_cost`: repo-native bounded-DP baseline
- `reference.published_optimality_gap_pct`: the Figure 9 gap labels stored for that instance
- `repo_optimality_gap_pct`: heuristic gaps reproduced by the current code against the bounded DP
- `repo_gap_minus_paper_pct`: difference between the reproduced heuristic gaps and the published gaps

Interpretation:

- use the Figure 9 gap labels to validate ranking and rough performance fidelity against the paper
- use the bounded DP only as the repo-native absolute baseline, not as a claim that the paper’s exact
  optimal costs were reproduced verbatim

Python package helpers for the Gijs experiment family:

- `get_benchmark_grid()`
- `build_grid_instances()`

## Current repo findings

Primary screening instance:

- `dual_l4_ce110`

Current learned-policy results on that instance:

- smoke run, oblique depth-2 linear-leaf soft tree: `249.84`
- full-budget run, oblique depth-2 linear-leaf soft tree: `233.08375`

Current heuristic baseline on the same full-budget evaluation protocol:

- single-index: `226.816875`
- dual-index: `222.4025`
- capped dual-index: `221.61`
- tailored base-surge: `222.7825`

So the current learned tree is materially better than the first smoke run, but still about `5.2%` worse
than the best heuristic on the primary literature instance.

Updated `l_r=3` diagnosis:

- `dual_l3_ce105` and `dual_l3_ce110` were not mainly failing because the tree could not see the right
  state; they were failing because the regular-order control surface was too loose
- the successful fix was to move from the wider oblique linear family to a tighter
  axis-aligned constant-leaf tree over a small-cap capped-dual-index control grid
- that design keeps the learned controls close to the benchmark heuristic structure and sharply
  improves both problematic `l_r=3` rows in current screening runs

## What we know about policy structure

The current direct vector-action tree uses the full reduced state and outputs raw order quantities:

- `(q_regular, q_expedited)`

The strong benchmark heuristics do not operate in that coordinate system. They operate on low-dimensional
inventory-position summaries:

- expedited inventory position
- regular inventory position

and then derive the final order quantities from target levels or caps.

This matters because the current tree search space must learn two things at once:

- how to compress the state into the relevant inventory-position coordinates
- how to map those coordinates into coupled source decisions

That is a plausible explanation for why the same soft-tree family transfers well to fixed-cost lost sales
but not yet to dual sourcing.

## Next policy direction

The repo has now moved beyond direct-order trees and plain target-position trees. The most promising
next step is to keep the small-cap capped-dual-index control family, but spend budget on the
better-behaved tree classes first:

- axis-aligned splits before oblique splits on the `l_r=3` rows
- constant-leaf or otherwise tightly discretized controls before wider linear leaves
- only after that, larger or more flexible policy classes

The evidence so far is that dual sourcing benefits more from tighter inductive bias around the
benchmark heuristic geometry than from simply increasing function-class flexibility.
