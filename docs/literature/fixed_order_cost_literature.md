# Lost Sales With Fixed Ordering Cost

This note captures the literature target for the next problem extension: a periodic-review single-item lost-sales system with a fixed setup cost whenever a positive order is placed.

## Core Reference

- Marco Bijvank, Sandjai Bhulai, Woonghee Tim Huh, "Parametric replenishment policies for inventory systems with lost sales and fixed order cost," *European Journal of Operational Research* 241(2):381-390, 2015.
- Author-hosted PDF: https://www.math.vu.nl/~sbhulai/publications/ejor2015b.pdf
- Review article for broader lost-sales context: https://www.sciencedirect.com/science/article/abs/pii/S0377221711001354
- Marco Bijvank, *Service Inventory Management: solution techniques for inventory systems without backorders*, PhD thesis, Vrije Universiteit Amsterdam, 2009.
- Dissertation PDF: https://research.vu.nl/ws/portalfiles/portal/42184715/complete%20dissertation.pdf

## Problem Variant

Relative to the current baseline repo problem, the extension adds a fixed ordering cost `K` whenever `Q_t > 0`. The remaining ingredients stay aligned with the current environment:

- periodic review
- single item
- lost sales
- positive lead time
- holding cost `h`
- lost-sales penalty `p`
- stochastic demand

This means the environment in `invman.env.lost_sales` can support the variant directly with `fixed_order_cost > 0`; the missing pieces are benchmark policy classes and experiment presets.

## Parameter Grid Used In The Reference Study

The 2015 EJOR paper evaluates 1080 instances over:

- lead time `L in {1, 2, 3, 4}`
- holding cost `h = 1`
- lost-sales penalty `p in {4, 9, 14, 19, 39, 99}`
- fixed ordering cost `K in {5, 10, 25, 50}`
- mean demand `mu in {2.5, 5, 10, 20}`
- demand variance-to-mean ratio `1` (Poisson) or `{2, 4}` (negative binomial)

The paper excludes the `L = 4, mu = 20` cases because the exact dynamic-programming benchmark becomes too expensive.

## Benchmark Algorithms Reported

The paper compares the following policy classes:

- exact optimal policy from dynamic programming
- classic `(s, S)` policy
- `(s, nQ)` policy
- modified `(s, S, q)` policy with an explicit maximum order quantity
- heuristic modified `(s, S, q_bar)` policy
- backorder-optimal `(s, S)` policy used as a transfer baseline

Aggregate average optimality gaps reported across the test bed:

- backorder `(s, S)`: `5.00%`
- `(s, S)`: `1.08%`
- `(s, nQ)`: `1.22%`
- modified `(s, S, q)`: `0.55%`
- heuristic modified `(s, S, q_bar)`: `0.82%`

## What The Literature Search Actually Found

After an extended search, the evidence splits into two tiers:

- The 2015 EJOR paper gives the large test bed, the policy classes, and aggregate optimality-gap summaries.
- The 2015 paper does **not** appear to publish a full per-instance table for the 1080-instance benchmark in the paper PDF.
- The 2015 paper does include one illustrative state-space example and several aggregate summary tables, but not the exact per-instance benchmark values we would want for our 16-instance subset.
- The 2009 dissertation does publish **exact per-instance numerical tables**, but for a related fixed-order-cost benchmark family rather than the exact 2015 grid.

So the repo should treat the literature as providing:

- aggregate benchmark targets from the 2015 paper
- exact related-instance anchor values from the 2009 dissertation
- repo-native benchmark values for the canonical subset grid computed by our own code

## Exact Numerical Anchors Found In The Dissertation

Chapter 6 of the 2009 dissertation reports exact-cost numerical tables for lost-sales systems with fixed order costs under:

- fixed penalty `p = 19`
- review period `R = 1`
- lead times `L in {0.5, 1.5, 2.5, 3.5}`
- fixed order costs `K in {25, 50, 100}`
- pure Poisson, compound Poisson, and negative-binomial demand families

This is not the same benchmark family as the 2015 paper, but it does provide published exact values for related instances. For example, Table 6.11 reports the following pure-Poisson exact-cost rows:

- `(lambda, mu, L, K) = (5, 1, 0.5, 25)` with optimal cost `20.43`
- `(lambda, mu, L, K) = (5, 1, 1.5, 50)` with optimal cost `27.34`
- `(lambda, mu, L, K) = (5, 1, 2.5, 25)` with optimal cost `22.34`
- `(lambda, mu, L, K) = (5, 1, 3.5, 25)` with optimal cost `22.90`

These dissertation tables are useful as literature validation anchors, but they are not a substitute for a benchmark table on the exact 2015 subset we encoded in this repo.

## Suggested Starter Preset For This Repo

To stay close to the current lost-sales experiments while keeping the benchmark small, start with:

- demand distribution: Poisson
- `mu = 5`
- `h = 1`
- `p in {4, 19}`
- `K in {5, 25}`
- `L in {1, 2, 3, 4}`

That preserves comparability with the current paper and keeps the first extension manageable before adding negative-binomial demand and richer benchmark policies.

## Canonical Benchmark Subset In This Repo

The codebase now treats the following 16-instance subset as the canonical first benchmark grid for the fixed-order-cost extension:

- demand distribution: Poisson
- `mu = 5`
- `h = 1`
- `p in {4, 19}`
- `K in {5, 25}`
- `L in {1, 2, 3, 4}`

This grid now lives in the Rust reference grid and Python benchmark glue exposed by
`scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py` (`FULL_GRID_NAME =
"lost_sales_style_full_grid_mu5"`). The historical Python module path
`invman.problems.lost_sales_fixed_order_cost.reference_instances` was removed in the Rust-first
cleanup.

The first repo-native baseline for this grid is stored in:

- `../benchmarks/fixed_order_cost_literature_subset_poisson_mu5.json`
- `../benchmarks/fixed_order_cost_literature_subset_poisson_mu5.md`

Using the current benchmark code and default search/evaluation settings, the repo baseline currently shows:

- modified `(s, S, q)` is not worse than `(s, S)` on `13 / 16` instances
- modified `(s, S, q)` is not worse than `(s, nQ)` on `12 / 16` instances
- mean relative improvement of modified `(s, S, q)` versus `(s, S)` is `0.67%`
- mean relative improvement of modified `(s, S, q)` versus `(s, nQ)` is `0.20%`
