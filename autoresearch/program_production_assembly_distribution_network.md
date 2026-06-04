# production-assembly-distribution-network autoresearch

This is the network counterpart to the dual-sourcing / multi-echelon / OWMR / JR / VMI
autoresearch programs. It targets the `production_assembly_distribution_network` problem
(Pirhooshyaran & Snyder 2021, arXiv:2006.05608): a finite-horizon stochastic multi-echelon
inventory MDP on a directed acyclic supply network with raw-material and finished-goods
inventories, pairwise order-up-to decisions on supply relations, and order-after-demand.

The single-policy loop has the same shape as the sibling programs: train ONE soft-tree
CMA-ES policy on the NAMED instance, evaluate its held-out paired-CRN cost + gap vs the
strongest in-env heuristic, append a TSV ledger row. The runner is
`scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py`;
it drives the binding `production_assembly_distribution_network_soft_tree_population_rollout`
directly (no Python re-implementation; mirrors the existing
`reproduce_pirhooshyaran_serial_case3.py` in that folder).

## HONEST STATUS — research env, not a literature reproduction

This env is FAITHFUL to the Pirhooshyaran & Snyder (2021) MDP (eq. 1-13, cost eq. 3,
verified equation-by-equation in-crate) but is **NOT literature-verified**: `literature_verified
= false` on every reference instance. There is **NO published optimum for THIS network env**.
The serial textbook optimum **47.65 is structurally UNREACHABLE here** — it is an *echelon*
base-stock level applied to this env's *local* raw-position pairwise policy (eq. 5, which
excludes finished goods), a level-interpretation mismatch documented in the env README; the
serial optimum's literature-verified home is the sibling `multi_echelon/serial` family, not
here. Only the single-node newsvendor row is literature-verified for this family.

Consequently the baseline here is a **research comparison, not a literature reproduction**:
the env's OWN best pairwise base-stock, grid-searched over the per-relation OUL levels. This
is stated honestly and must NOT be dressed up as an optimum.

## Benchmark

The trusted instance is the env's `PRIMARY_REFERENCE_INSTANCE`,
`pirhooshyaran2021_serial_case3` (Tables 2-3, serial case 3 = Snyder & Shen Example 6.1):

- 3 nodes, serial 0->1->2; node 0 the only source.
- external customer demand N(5,1) at the downstream node (node 2) only; T = 10.
- shipment lead times: external->0 = 2, edge 0->1 = 1, edge 1->2 = 1.
- local holding costs (upstream->downstream) [2, 4, 7]; backorder cost 37.12 at node 2 only.
- supply relations (env order = edges first, then external suppliers):
  relation 0 = edge(0->1), relation 1 = edge(1->2), relation 2 = external->node 0.
- initial state: finished [10,5,5], pipelines [[0],[0],[0,0]], everything else 0.
- undiscounted average per-period cost (matches the paper's average-cost comparison).

Strongest in-env heuristic = **best pairwise base-stock** (`pairwise_base_stock` policy via
`production_assembly_distribution_network_policy_rollout_from_paths`), found by grid-searching
the per-relation OUL levels on a disjoint search block (256 paths, seed 500_000) and re-scoring
the argmin on the held-out block. THIS is the keep/discard gate.

- best pairwise OUL ≈ **(8, 7, 9)** (relations [edge0, edge1, external]) — the env's own
  argmin, NOT the carried analytical levels.
- the carried analytical Clark-Scarf levels under mapping A ([5.53, 6.49, 10.69]) cost ~68
  per period here (worse): echelon levels are the wrong local targets for this env.
- held-out best pairwise base-stock ≈ **59.65 ± 0.39** per period (2000-path block).

Evaluation protocol: disjoint search (seed 500_000) and held-out (seed 900_000) demand-path
blocks; demand only at node 2, N(5,1) rounded/clipped, T = 10, undiscounted. The SAME held-out
block scores the learned soft-tree (via `..._soft_tree_rollout_from_paths`) and the pairwise
base-stock (via `..._policy_rollout_from_paths`) — paired / variance-reduced.

## Action design — the contribution

The soft-tree rollout binding emits a **direct order quantity per supply relation**
(`vector_quantity`, action_dim = supply_relation_count = 3) clipped to [0, 60]. This is the
analogue of OWMR's weak `direct_orders` baseline: a CONSTANT leaf can only emit a fixed order
rate and cannot react to inventory, so it cannot express order-up-to behavior. The lever is
the **leaf class**, not the optimizer budget. A **LINEAR leaf** maps the (scaled) policy-state
features — which INCLUDE per-relation raw inventory and per-relation in-transit pipeline — to
the per-relation order, so it can express inventory-position feedback (the q = level −
max(IP,0) shape that base-stock targets) and oblique splits let it switch behavior by
inventory regime, expressing a richer-than-base-stock response. The env owns its policy input
dimension (30 for case3); we ask the binding rather than re-deriving it.

Warm start: the direct-quantity decoder is not analytically invertible into a base-stock
encoding (features are divided by a dynamic per-step scale), so we use honest decoder-agnostic
anchoring — seed the CMA mean at the steady-state **flow rate** (order the demand mean ≈ 5 per
relation each period; the linear leaf bias = softplus_inv(flow), leaf weights = 0). Generation
0 reproduces a sensible flow policy (~70/period) and CMA-ES refines outward toward the
inventory-feedback regime.

## Search surface (editable levers)

- soft-tree structure: `--depth` (2,3), `--temperature`, `--split_type`
  (oblique / axis_aligned), `--leaf_type` (constant / linear / sigmoid_linear).
- action box cap `MAX_VALUES` (currently 60, well above the operating region ~5-25).
- CMA-ES warm-start flow rate (`--warm_start_flow`, default = demand mean).
- budget: `smoke` (validate only), `screening`, `full`.

## Autoresearch outcome (RESULT)

Headline (full budget; depth-2 oblique LINEAR-leaf soft tree, 465 params, temperature 0.25,
`vector_quantity` action over the 3 supply relations, warm-started at flow=5; CMA-ES
popsize 24, generations 60, train_seed_batch 96, paired CRN; held-out 4000 paths;
commit 01c657a; ~89 s train on 2 rayon cores):

- best pairwise base-stock (held-out, OUL = [8, 7, 9]): **60.24 / period** (research baseline,
  NOT an optimum). On the 2000-path screening block it locks at 59.65 ± 0.39; the difference is
  CRN block variance.
- learned soft-tree (held-out): **57.25 ± 0.22 / period** (gen-0 flow warm-start was 70.85).
- gap: **−2.99 cost / −4.96 % (learned BEATS the env's own best pairwise base-stock)**, robustly
  outside the held-out stderr (~0.22), and **reproduced across CMA seeds and depth {2,3}**:
  depth2-seed123 −4.0 %, depth2-seed321 −7.9 %, depth3-seed123 −3.0 % (every config wins by
  >3 % at ~0.2-0.25 stderr).

WHY it beats the heuristic: the pairwise base-stock policy uses LOCAL raw-position feedback
only; the learned linear-leaf direct-quantity policy can additionally read finished inventory,
internal/external backlog, and inbound pipeline per node, and switch order behavior by
inventory regime (oblique splits) — a strictly richer control class on the SAME action
relations. This is the same "action design / leaf class, not capacity, is the lever" thesis as
the OWMR `direct_orders`→structured and multi-echelon grid→`direct_level` flips, here on a
faithful-but-non-literature-verified network MDP.

Constant-leaf direct-quantity trees stay at the flow regime and lose to the heuristic (a fixed
order rate cannot express base-stock), confirming the leaf-class lever.

This is a RESEARCH result on a non-literature-verified env: it shows the learned policy beats
the env's own best pairwise base-stock, NOT that it reproduces or beats any published cost.
