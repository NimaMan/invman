# dual-sourcing autoresearch

This is the dual-sourcing counterpart to the lost-sales and fixed-cost autoresearch programs.

## Benchmark

Primary screening instance:

- `dual_l4_ce110`
- regular lead time `l_r = 4`
- expedited lead time `l_e = 0`
- demand uniform on `{0,1,2,3,4}`
- `h = 5`
- `b = 495`
- `c_r = 100`
- `c_e = 110`

The benchmark heuristics are fixed:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

## Intended search surface

- `invman/policies/`
- `rust/src/policies/`
- `rust/src/rollout/`
- limited support code needed to wire vector-action trees into training

## Budgets

Use the budgets from `scripts/dual_sourcing/autoresearch_dual_sourcing.py`:

- `screening`
- `full`

## Goal

Lower the learned-policy cost on the primary dual-sourcing instance while preserving a clean
general policy pipeline.

Current smoke baseline:

- learned tree: `249.84`
- best heuristic baseline: capped dual-index `220.73`

Current full-budget baseline:

- learned tree: `233.08375`
- single-index: `226.816875`
- dual-index: `222.4025`
- capped dual-index: `221.61`
- tailored base-surge: `222.7825`

## What we know

The current direct vector-action soft tree is no longer just a smoke-test artifact. With a full budget,
it improves a lot versus the original smoke run, but it still remains clearly behind the best heuristics.

That suggests the next dual-sourcing search should not focus first on more CMA-ES budget or deeper trees.
It should focus on the policy output space.

The benchmark heuristics all work with inventory-position targets or related low-dimensional controls:

- expedited inventory position
- regular inventory position
- optional regular cap or regular base-surge quantity

The current learned tree instead outputs direct raw orders:

- `(q_regular, q_expedited)`

So it has to discover both the right state compression and the right replenishment logic in one search
space. That is a plausible bottleneck.

## Next autoresearch target

The next family to add and test is a learned target-position policy for dual sourcing:

- the policy outputs state-dependent expedited and regular targets
- a deterministic mapper converts those targets into `(q_regular, q_expedited)`
- optional extension: a third output for a regular cap

This remains a learned, state-dependent policy class, but searches in a coordinate system that matches
the strongest known heuristic families much better than direct raw order quantities.
