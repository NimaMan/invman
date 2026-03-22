# Policies

This package is the canonical home for learned policy parameterizations.

Current policy families:

- `LinearPolicyNet`: linear policy with categorical or ordinal quantity heads
- `PolicyNet`: small MLP policy with categorical or ordinal quantity heads
- `SoftTreePolicy`: soft decision tree with configurable split and leaf structure

## Current benchmark findings

Trusted benchmark:

- vanilla lost sales
- `L=4`
- shortage cost `p=4`
- demand `~ Poisson(5)`
- holding cost `h=1`

Known reference points on this instance:

- optimal reference: about `4.73`
- Myopic-2: about `4.82`
- linear learned policy: `4.8066`

Best learned policy found so far in this package:

- `soft_tree_oblique_tree_linear_leaf_quantity_pipeline`
- depth `2`
- full-budget mean cost: `4.753725`

This beats the heuristic baseline `Myopic-2 = 4.8204` and improves on the earlier constant-leaf
soft-tree result `4.7980`.

## Tree-policy findings

From the completed tree-structure search on the vanilla benchmark:

- oblique splits worked much better than axis-aligned splits
- shallow trees worked better than deeper trees under the same protocol
- richer leaves were the high-value change

The winning structure so far is:

- oblique splits
- depth `2`
- linear leaf outputs

## Open direction

The next intended transfer target is the fixed-order-cost lost-sales problem, where current learned
policies still lag the benchmark heuristics.
