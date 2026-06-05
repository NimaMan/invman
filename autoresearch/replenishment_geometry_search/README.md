# Replenishment Geometry Search

This note tracks the current architecture study for the `L=4`, `p=4` lost-sales family.
The goal is to understand which policy parameterizations are easy enough for CMA-ES to search,
and how that answer changes between:

- vanilla lost sales
- fixed-cost lost sales
- different demand families with the same mean demand

The current focus is linear policies and soft trees. We are not trying to settle the final
benchmark surface here; we are trying to understand what kinds of replenishment maps are easy to
find.

## Current Protocol

Unless stated otherwise, the current single-instance runs use:

- `training_episodes = 2000`
- `es_population = 64`
- `training_horizon = 2000`
- `eval_horizon = 10^6`
- `eval_seeds = 10`
- seed `42`

Demand cases used in the current demand-robustness sweep:

- `Poisson`, mean demand `5`
- `Geometric`, mean demand `5`
- positive `MarkovModulatedPoisson2`, mean-preserving:
  - `lambda_low = 3`
  - `lambda_high = 7`
  - `p00 = p11 = 0.9`
- negative `MarkovModulatedPoisson2`, mean-preserving:
  - `lambda_low = 3`
  - `lambda_high = 7`
  - `p00 = p11 = 0.1`

These demand extensions are implemented in:

- [mod.rs](/home/nima/code/ml/invman/src/problems/lost_sales/demand/mod.rs)

## Policy Equations

Let `z = w^T x + b` denote the scalar logit of a linear policy, and let `Q_env` be the environment
action cap.

The policy family currently under study is:

| Policy | Quantity map |
| --- | --- |
| `linear_categorical_quantity` | `a = argmax_k z_k` |
| `linear_sigmoid_direct_quantity` | `a = round(Q_env * sigmoid(z))` |
| `linear_direct_quantity` | `a = clip(round(softplus(z)), 0, Q_env)` |
| `linear_gated_sigmoid_direct_quantity` | `a = round(sigmoid(g) * sigmoid(q) * Q_env)` |
| `linear_soft_gated_direct_quantity` | `a = clip(round(sigmoid(g) * softplus(q)), 0, Q_env)` |
| `linear_hard_gated_direct_quantity` | `a = 0` if `sigmoid(g) < 0.5`, else `clip(round(softplus(q)), 1, Q_env)` |
| `linear_soft_gated_ordinal_quantity` | `a = round(sigmoid(g) * sum_{k=1}^{Q_env} sigmoid(o_k))` |
| `soft_tree_depth1_linear_leaf` | `a = clip(round(sum_l pi_l(x) * softplus(alpha_l^T x + beta_l))), 0, Q_env)` |
| `soft_tree_depth2_linear_leaf` | same as above with depth `2` routing |

Two important distinctions:

- `sigmoid_direct_quantity` is still `Q`-dependent inside the head.
- `direct_quantity` is `Q`-free inside the head; `Q_env` only appears in the final projection.

The visualization script for the scalar head maps lives at:

- [visualize_linear_head_geometry.py](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/visualize_linear_head_geometry.py)

and generates:

- [scalar_head_shapes_q20.svg](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/artifacts/scalar_head_shapes_q20.svg)
- [scalar_head_shapes_q20.png](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/artifacts/scalar_head_shapes_q20.png)
- [scalar_head_action_bins_q20.json](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/artifacts/scalar_head_action_bins_q20.json)

## Canonical Poisson Results

The cleanest comparison is still the canonical Poisson case:

- vanilla: `vanilla_l4_p4_poisson5`
- fixed cost: `lit_pois_mu5_l4_p4_k5`

### Vanilla Lost Sales, Poisson

Results from:

- [lost_sales_l4_p4_demand_policy_suite.json](/home/nima/code/ml/invman/outputs/benchmarks/lost_sales_l4_p4_demand_policy_suite_pop64_seed42/lost_sales_l4_p4_demand_policy_suite.json)

Heuristic baselines:

- `myopic1 = 5.06266`
- `myopic2 = 4.81677`
- `svbs = 5.83442`

Learned policies:

| Policy | Mean cost | Std. dev. |
| --- | ---: | ---: |
| `linear_categorical_quantity` | 4.87516 | 0.00865 |
| `linear_sigmoid_direct_quantity` | 4.75568 | 0.00727 |
| `linear_direct_quantity` | 4.74666 | 0.00746 |
| `linear_gated_sigmoid_direct_quantity` | 4.74505 | 0.00816 |
| `linear_soft_gated_direct_quantity` | 4.74807 | 0.00800 |
| `linear_hard_gated_direct_quantity` | 4.74797 | 0.00760 |
| `linear_soft_gated_ordinal_quantity` | 4.75027 | 0.00769 |
| `soft_tree_depth1_linear_leaf` | 4.74812 | 0.00748 |
| `soft_tree_depth2_linear_leaf` | 4.75182 | 0.00783 |

Takeaway:

- categorical is clearly worse
- everything else is tightly clustered in the good regime
- under `pop64`, `linear_sigmoid_direct_quantity` is competitive on vanilla

This is important because it corrects an earlier overstatement: the one-head sigmoid direct policy
is not fundamentally bad on vanilla. It is search-sensitive at smaller populations.

### Fixed-Cost Lost Sales, Poisson

Results from:

- [fixed_cost_l4_p4_k5_demand_policy_suite.json](/home/nima/code/ml/invman/outputs/benchmarks/fixed_cost_l4_p4_k5_demand_policy_suite_pop64_seed42/fixed_cost_l4_p4_k5_demand_policy_suite.json)

Heuristic baselines:

- `(s,S) = 9.36774`
- `(s,nQ) = 9.20936`
- `modified (s,S,q) = 9.20032`

Learned policies:

| Policy | Mean cost | Std. dev. |
| --- | ---: | ---: |
| `linear_categorical_quantity` | 10.27542 | 0.00906 |
| `linear_sigmoid_direct_quantity` | 8.77280 | 0.00775 |
| `linear_direct_quantity` | 9.74819 | 0.00719 |
| `linear_gated_sigmoid_direct_quantity` | 9.74745 | 0.00737 |
| `linear_soft_gated_direct_quantity` | 8.77148 | 0.00684 |
| `linear_hard_gated_direct_quantity` | 9.74779 | 0.00720 |
| `linear_soft_gated_ordinal_quantity` | 8.77832 | 0.00741 |
| `soft_tree_depth1_linear_leaf` | 8.77083 | 0.00754 |
| `soft_tree_depth2_linear_leaf` | 8.77001 | 0.00759 |

Takeaway:

- the fixed-cost problem is much more sensitive
- `linear_direct_quantity` is not robust here under the current `2k / pop64` protocol
- good fixed-cost Poisson policies are:
  - `linear_sigmoid_direct_quantity`
  - `linear_soft_gated_direct_quantity`
  - `linear_soft_gated_ordinal_quantity`
  - the trees

So the current fixed-cost story is not “Q-free direct wins.” The stronger current statement is:

- some parameterizations make the fixed-cost replenishment geometry easy to search
- some do not

## Demand-Robustness Sweep

### Fixed-Cost Sweep: Complete

Full fixed-cost sweep root:

- [fixed_cost_l4_p4_k5_demand_policy_suite_pop64_seed42](/home/nima/code/ml/invman/outputs/benchmarks/fixed_cost_l4_p4_k5_demand_policy_suite_pop64_seed42)

Best policy by demand case:

| Demand case | Best learned policy | Mean cost | Best heuristic | Mean cost |
| --- | --- | ---: | --- | ---: |
| `Poisson` | `soft_tree_depth2_linear_leaf` | 8.77001 | `modified (s,S,q)` | 9.20032 |
| `Geometric` | `linear_hard_gated_direct_quantity` | 13.62760 | `modified (s,S,q)` | 14.00227 |
| positive `MMPP2` | `linear_hard_gated_direct_quantity` | 11.87871 | `modified (s,S,q)` | 12.09351 |
| negative `MMPP2` | `linear_soft_gated_direct_quantity` | 9.06558 | `modified (s,S,q)` | 9.65013 |

This is the clearest architecture result we currently have:

- there is no single winner across all fixed-cost demand families
- the best head depends on the demand geometry
- direct, gated-direct, ordinal, and tree policies are all useful candidates

More specifically:

- positively correlated and overdispersed fixed-cost demand favor harder thresholded direct heads
- negatively correlated fixed-cost demand favors the soft-gated direct head
- Poisson fixed-cost demand still slightly favors trees

### Vanilla Sweep: In Progress

Vanilla sweep root:

- [lost_sales_l4_p4_demand_policy_suite_pop64_seed42](/home/nima/code/ml/invman/outputs/benchmarks/lost_sales_l4_p4_demand_policy_suite_pop64_seed42)

Completed so far:

| Demand case | Current best learned policy | Mean cost | Best heuristic | Mean cost | Status |
| --- | --- | ---: | --- | ---: | --- |
| `Poisson` | `linear_gated_sigmoid_direct_quantity` | 4.74505 | `myopic2` | 4.81677 | complete |
| `Geometric` | `soft_tree_depth1_linear_leaf` | 10.63912 | `myopic2` | 10.80254 | partial: `soft_tree_depth2_linear_leaf` still evaluating |
| positive `MMPP2` | pending | — | — | — | not started yet |
| negative `MMPP2` | pending | — | — | — | not started yet |

Current vanilla read:

- vanilla is much less architecture-sensitive than fixed cost
- once we move away from categorical, almost every direct/tree policy lands in the same good basin
- the main thing we have not finished yet is whether correlated vanilla demand breaks that picture

## What We Know Right Now

The current local facts are:

1. `linear_unbounded_direct_quantity` is still the bad policy family.
   - it remains excluded from active experiments
   - the problem is not expressivity; it is a bad search parameterization

2. `linear_sigmoid_direct_quantity` is not fundamentally bad.
   - on vanilla Poisson it becomes good at `pop64`
   - earlier bad vanilla results were mainly a search-budget issue

3. The fixed-cost problem is still the stricter test.
   - different heads win under different demand families
   - the current best family is “structured direct or tree,” not one single architecture

4. The vanilla problem is easier.
   - categorical is still weak
   - but most non-categorical direct/tree heads are tightly clustered once the population is large enough

5. The main question is still search geometry.
   - this note is not evidence that one head is universally optimal
   - it is evidence that some heads expose the replenishment map more cleanly to CMA-ES than others

## Why The Sigmoid-vs-Softplus Discussion Still Matters

The earlier shape discussion was directionally useful, but too strong when it was phrased as a hard
success/failure claim.

The better current interpretation is:

- `softplus` gives a more search-friendly positive-part geometry
- `Q * sigmoid(z)` gives a more compressed and population-sensitive geometry
- that does **not** make sigmoid direct impossible
- it makes it less robust at smaller CMA populations

This is exactly what the pop-size probes showed on vanilla Poisson:

- `pop50`: `linear_sigmoid_direct_quantity` failed badly
- `pop64`: it recovered the good basin
- `pop200`: it also recovered the good basin

So the shape argument is now a search-robustness argument, not an expressivity argument.

## Verification Example

We keep one published verification anchor in the repo:

- reference instance: `bijvank2015_table1_l2_p14_k5`

This is taken from:

- Marco Bijvank, Sandjai Bhulai, Woonghee Tim Huh (2015),
  *Parametric replenishment policies for inventory systems with lost sales and fixed order cost*,
  *European Journal of Operational Research*, 241(2):381-390.

The repo encodes this validation case in:

- [references.rs](/home/nima/code/ml/invman/src/problems/lost_sales/fixed_order_cost/literature/references.rs)
- [validate_known_optimum.py](/home/nima/code/ml/invman/scripts/lost_sales_fixed_order_cost/validate_known_optimum.py)

The paper-aligned parameters are:

- review-period demand mean `5`
- Poisson demand
- lead time `L = 2`
- shortage cost `p = 14`
- fixed order cost `K = 5`
- holding cost `h = 1`

Published values recorded in the repo:

| Quantity | Value |
| --- | ---: |
| reported optimum | 11.46 |
| reported `(s,S)` | 11.62 |
| reported `(s,nQ)` | 11.56 |
| reported modified `(s,S,q)` | 11.50 |

Reported heuristic parameters:

- `(s,S) = (17, 23)`
- `(s,nQ) = (17, 7)`
- modified `(s,S,q) = (17, 23, 7)`

This validation example is not the main architecture benchmark. It is the literature-backed
correctness anchor we use to make sure the fixed-cost environment and heuristic evaluation still
match the published reference family.

## Files

Main experiment outputs referenced in this note:

- [lost_sales_l4_p4_demand_policy_suite.json](/home/nima/code/ml/invman/outputs/benchmarks/lost_sales_l4_p4_demand_policy_suite_pop64_seed42/lost_sales_l4_p4_demand_policy_suite.json)
- [fixed_cost_l4_p4_k5_demand_policy_suite.json](/home/nima/code/ml/invman/outputs/benchmarks/fixed_cost_l4_p4_k5_demand_policy_suite_pop64_seed42/fixed_cost_l4_p4_k5_demand_policy_suite.json)
- [visualize_linear_head_geometry.py](/home/nima/code/ml/invman/autoresearch/replenishment_geometry_search/visualize_linear_head_geometry.py)
