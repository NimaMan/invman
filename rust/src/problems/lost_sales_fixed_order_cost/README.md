# Fixed-Cost Lost Sales

This package is the fixed-order-cost extension of the single-item lost-sales problem.

## Formulation

The problem follows the periodic-review lost-sales model of Bijvank, Bhulai, and Huh (2015):

- one item
- periodic review with review period `R`
- deterministic lead time `L = lR`
- i.i.d. demand per review period
- lost sales for unmet demand
- linear holding cost `h`
- linear lost-sales penalty `p`
- fixed order cost `K` whenever the order quantity is positive

At a review instant after order delivery but before ordering, the state is `(i, y)`:

- `i`: on-hand inventory
- `y`: outstanding orders that will arrive over the next `l-1` review periods

For the published validation instance in Bijvank et al. Table 1:

- `R = 1`
- `L = 2`
- `h = 1`
- `p = 14`
- `K = 5`
- demand `~ Poisson(5)`

The expected one-period holding and lost-sales cost is

- `c(i) = h E[(i - D)^+] + p E[(D - i)^+]`

and the paper evaluates long-run average cost via value iteration.

## Literature Anchor

Primary source:

- Bijvank, Bhulai, and Huh (2015), *Parametric replenishment policies for inventory systems with lost sales and fixed order cost*
- URL: <https://www.math.vu.nl/~sbhulai/publications/ejor2015b.pdf>

Published Table 1 validation row:

- optimal average cost: `11.46`
- best `(s,S)`: `s=17, S=23`, cost `11.62`
- best `(s,nQ)`: `s=17, q=7`, cost `11.56`
- best modified `(s,S,q)`: `s=17, S=23, q=7`, cost `11.50`

## Current Rust Status

This Rust package now contains:

- exact average-cost value iteration for the published Poisson validation instance
- exact evaluation of the published `(s,S)`, `(s,nQ)`, and modified `(s,S,q)` policies
- demand-path rollout/search helpers for simulation-backed heuristic work

The exact verifier uses a bounded inventory-position state space and matches the published Table 1
row tightly once the inventory-position cap is at least `24`.

Current Rust reproduction at cap `24`:

- optimal: `11.463052`
- `(s,S)`: `11.618148`
- `(s,nQ)`: `11.555216`
- modified `(s,S,q)`: `11.497403`

These are within about `0.005` of the published numbers, so this validation instance is currently
treated as literature-verified.

## Scope

What is literature-verified here:

- the published Bijvank Table 1 validation instance
- the Rust exact solver and exact heuristic evaluators on that instance

What is not yet literature-verified here:

- the larger fixed-cost benchmark grids used elsewhere in the repo
- learned-policy results on those grids

## Structure

- `references.rs`: literature source and published validation row
- `exact_value_iteration.rs`: exact average-cost solver and exact heuristic evaluation
- `heuristics.rs`: demand-path rollout and search helpers
- `bindings.rs`: Python bindings for the literature summary and heuristic tools
