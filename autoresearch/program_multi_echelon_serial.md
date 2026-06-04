# program_multi_echelon_serial — autoresearch for the serial Clark-Scarf env

The objective of this program is one honest learned-policy result on the
literature-verified **serial multi-echelon** env (`rust/src/problems/multi_echelon/serial`),
the textbook Clark-Scarf model (Clark & Scarf 1960; Snyder & Shen *Fundamentals of
Supply Chain Theory* Example 6.1). It is a **MATCH-only** problem: the comparator is a
TRUE optimum, so the honest ceiling is to reproduce it, never to beat it.

## Trusted benchmark (fixed)

- Instance: **Snyder & Shen Example 6.1** — 3 stages, Normal(5,1) demand, lead times
  (upstream→downstream) [2,1,1], echelon holding [2,2,3], penalty 37.12. In the env's
  downstream→upstream convention: installation holding [7,4,2], echelon holding [3,2,2],
  lead [1,1,2].
- Baseline = the **Clark-Scarf OPTIMUM 47.65** (the one published anchor). The exact
  recursive-newsvendor solver returns 47.6654; the env simulation under the exact echelon
  base-stock levels reproduces 47.65 to **+0.06%** with continuous Normal demand (the env
  drops demand rounding). The optimal echelon base-stock policy is the optimal policy
  CLASS, so a learned policy can at best TIE.
- Env reproduction is verified in `serial/verification.rs` (exact-vs-Ex6.1 literature-
  verified; Poisson reference-implementation-verified; env-sim self-consistent at
  downstream lead time 1, which Ex6.1 satisfies).

## The binding (call-bridge added)

- `multi_echelon_serial_soft_tree_population_rollout`
  (`rust/src/problems/multi_echelon/serial/bindings.rs`) — decodes a soft-tree policy
  into the serial decision and rolls out `echelon_base_stock.rs`-style dynamics in Rust
  (`serial/rollout.rs`), returning per-individual mean per-period cost under paired CRN.
- `multi_echelon_serial_exact_normal_solution` — exact Clark-Scarf solver helper, used to
  warm-start the policy at the optimum and to report the MATCH baseline.
- Registered via `serial/bindings.rs::register_py` → `multi_echelon/bindings.rs`.

## Action geometry (the policy) — `direct_level`

The serial decision class is **echelon base-stock**: each stage k orders
`max(0, S_k − echelon_IP_k)`. The policy emits the N echelon base-stock LEVELS directly
(continuous, non-negative, bounded by a generous physical ceiling), via the new continuous
soft-tree head `action_vector_continuous_from_flat_params` in
`rust/src/core/policies/soft_tree.rs`. It lives in the optimal policy's coordinate system,
so warm-starting the constant leaves at the exact Clark-Scarf levels makes **generation 0
reproduce the optimum** (verified: 47.68, the +0.06% env-sim band). The warm-start anchor
is always kept in the candidate set, so the reported policy can never be worse than the
gen-0 anchor (a true optimum cannot be beaten).

## Optimization protocol

CMA-ES warm-started at the encoded exact levels (small sigma), scored by the population
rollout under paired CRN; the incumbent and a generation candidate are re-evaluated on a
held-out CRN block at full periods. Budgets in
`scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py` (`smoke` / `screening`
/ `full`).

## What we know (autoresearch outcome)

- The constant-leaf warm start at the exact echelon levels reproduces the Clark-Scarf
  optimum to within the env's own +0.06% reproduction band (gen-0 ≈ 47.67 vs published
  47.65). The learned policy is reported as a **match %** = 100·47.65/learned_cost and a
  signed gap to the optimum; we never claim to beat the optimum.
- Lever: `direct_level` echelon-level estimation + warm-start at the exact solution. The
  optimizer searches outward from the optimum; on this convex serial instance the best it
  can do is tie within sampling error.

## Result (committed run, `outputs/` is gitignored)

| budget | learned_cost | published_optimum | match % | gap vs published | verdict |
|---|---|---|---|---|---|
| smoke | 48.378 | 47.65 | 98.49% | +1.53% | above_optimum (under-converged) |
| full | **47.6554** | 47.65 | **99.99%** | +0.011% | **matches_optimum** |

The warm-started Clark-Scarf direct-level soft tree ties the true optimum to within the
env's own +0.06% reproduction band — the honest ceiling for a proven-optimal comparator.
We never claim to beat it.

Runner: `scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py`.
Ledger / JSON: `outputs/autoresearch/multi_echelon_serial_autoresearch/`.
