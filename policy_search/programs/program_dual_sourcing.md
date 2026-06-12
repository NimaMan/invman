# dual-sourcing autoresearch

This is the dual-sourcing counterpart to the lost-sales and fixed-cost autoresearch programs.

## Benchmark

Use the full six-row Gijs Figure 9 family as the trusted design set:

- `dual_l2_ce105`
- `dual_l2_ce110`
- `dual_l3_ce105`
- `dual_l3_ce110`
- `dual_l4_ce105`
- `dual_l4_ce110`

Shared parameters:

- regular lead time `l_r in {2,3,4}`
- expedited lead time `l_e = 0`
- demand uniform on `{0,1,2,3,4}`
- `h = 5`
- `b = 495`
- `c_r = 100`
- `c_e in {105,110}`

Benchmark heuristics:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

## Intended search surface

- `invman/policy.py`
- `invman/policy_registry.py`
- `invman/policy_build.py`
- `invman/rollout_fitness.py`
- `src/problems/dual_sourcing/`
- `scripts/dual_sourcing/`
- `policy_search/studies/dual_sourcing_policy_search/`

## Budgets

Use the budgets in:

- `scripts/dual_sourcing/autoresearch_dual_sourcing.py`
- `policy_search/studies/dual_sourcing_policy_search/run_factor_screen.py`

The six-row factor screen is the first pass. Promoted candidates can then move to larger budgets.

## Goal

Use the six-row benchmark family to answer two questions:

1. which policy-design factors matter most?
2. what policy families are worth promoting to larger-budget searches?

Primary metric:

- relative gap to the best heuristic on the same benchmark row

Do not lock the search to one policy class. The job is to find very strong policies, not to prove that
soft trees are always best.

## What we know

The important shift is that dual sourcing is no longer a one-instance smoke-test problem.

Across the current six-row benchmark family, the evidence so far is:

- the dominant bottleneck is policy design, not missing simulator fidelity
- control geometry matters more than raw parameter count
- factorized dual-index controls work better than staying in raw order quantities
- a small discrete regular-cap grid helps because it keeps the learned surface close to the strongest heuristic family
- tighter trees can outperform wider oblique trees on the hard `l_r=3` and `l_r=4` rows

The current follow-up conclusion is more specific:

- `l_r = 2` rows improve with axis-aligned linear-leaf policies on the factorized capped-delta surface
- `l_r in {3,4}` still prefer the tighter axis-constant small-cap tree
- the best next design is a row-conditioned family or mixture on top of the same factorized control basis, not one universal policy geometry

So the next search should focus first on the policy/control surface, not just on more CMA-ES budget.

## Canonical workspace

The organized search surface now lives in:

- `policy_search/studies/dual_sourcing_policy_search/README.md`

Use that folder for:

- canonical factor screens across all six rows
- summaries of which design factors help most
- next-round candidate policies to promote
