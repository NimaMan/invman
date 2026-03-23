# Dual Sourcing

This package implements the small-scale dual-sourcing benchmark family used by Gijsbrechts et al. (2022) for the Veeraraghavan-Scheller-Wolf settings:

- regular supplier lead time `lr in {2, 3, 4}`
- expedited lead time `le = 0`
- discrete uniform demand on `{0, 1, 2, 3, 4}`
- holding cost `h = 5`
- backlog cost `b = 495`
- regular cost `c_r = 100`
- expedited cost `c_e in {105, 110}`

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

The literature does not provide a clean per-instance table of exact costs for all six settings, so exact
costs in this repo are repo-native benchmark values on the published problem family.

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

The next policy family to try here should keep the policy learned and state dependent, but change the
output space away from direct order quantities.

The best next candidate is a target-position policy:

- the learned policy outputs target expedited and regular positions, and optionally a regular cap
- a deterministic action mapper converts those targets into `(q_regular, q_expedited)`

This is more structured than the current direct vector-action tree, but still more general than the
constant-parameter benchmark heuristics because the targets can be state dependent and learned.
