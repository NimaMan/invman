# Policies

This package is the canonical home for learned policy parameterizations.

Current policy families:

- `LinearPolicyNet`: linear policy with categorical or ordinal quantity heads
- `PolicyNet`: small MLP policy with categorical or ordinal quantity heads
- `SoftTreePolicy`: soft decision tree with configurable split and leaf structure, now supporting
  scalar, vector, and discrete-grid action specifications

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

Fresh refresh runs after the Rust/problem-package refactor:

- Rust-backed soft tree rerun: `4.7658`
- fresh linear rerun: `5.0049`
- fresh NN `8x8` smoke rerun: `5.2504`

These refresh runs confirm the Rust soft-tree path is still sound. The linear and NN reruns were weaker
than their locked historical references, so the canonical benchmark anchors for those backbones remain
the older saved runs and paper values rather than the fresh smoke numbers.

## Tree-policy findings

From the completed tree-structure search on the vanilla benchmark:

- oblique splits worked much better than axis-aligned splits
- shallow trees worked better than deeper trees under the same protocol
- richer leaves were the high-value change

The winning structure so far is:

- oblique splits
- depth `2`
- linear leaf outputs

## Fixed-Cost Result

On the canonical fixed-order-cost instance `L=4, p=4, K=5, demand ~ Poisson(5)`:

- earlier transferred tree, `1M` eval: `8.81009`
- autoresearch-refined tree, `1M` eval: `8.76576`
- `s,S`: `9.44401`
- `s,nQ`: `9.21664`
- modified `s,S,q`: `9.16537`

The best fixed-cost tree is currently:

- oblique splits
- depth `1`
- linear leaf outputs

That is slightly shallower than the current vanilla winner. So the policy family is stable across
problems, but the best tree depth is not.

## Open Direction

The fixed-cost autoresearch result is now also positive on the canonical instance:

- best learned fixed-cost tree, `1M` eval: `8.76576`
- best heuristic, `1M` eval: `9.16537`

The next natural question is whether the same fixed-cost search loop can improve the broader
literature subset grid, not just the canonical instance.

## New Vector-Action Problems

The same soft-tree family now also supports:

- dual sourcing with 2D actions `(q_regular, q_expedited)`
- multi-echelon state-dependent base-stock actions `(y_w, y_r)`

Current smoke results:

- dual sourcing primary instance: learned tree `249.84`, best heuristic `220.73`
- multi-echelon setting 2: learned tree `3776.45`, best constant base-stock benchmark `3776.45`

## Dual-Sourcing Policy Finding

The current best tested learned policy on the primary dual-sourcing instance `dual_l4_ce110` is still
the direct vector-action soft tree:

- `soft_tree_oblique_tree_linear_leaf_quantity_pipeline`
- depth `2`
- full-budget mean cost: `233.08375`

Reference heuristics on the same evaluation:

- single-index: `226.816875`
- dual-index: `222.4025`
- capped dual-index: `221.61`
- tailored base-surge: `222.7825`

So the current learned tree is about `5.2%` worse than the best heuristic.

This is the first clear case in the repo where the current soft-tree family is probably not failing from
insufficient training budget alone. The more likely issue is the action representation:

- the current tree emits direct raw orders `(q_regular, q_expedited)`
- strong dual-sourcing heuristics operate in target-position space and only then derive order quantities

That makes dual sourcing the next policy-design problem rather than the next budget-tuning problem.

The next policy family to add here should therefore be a learned target-position policy, likely with
state-dependent outputs for expedited and regular targets instead of direct order quantities.
