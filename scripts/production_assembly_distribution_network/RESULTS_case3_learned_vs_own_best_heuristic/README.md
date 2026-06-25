# Results — production_assembly_distribution_network, case3 (refreshed 2026-06-04)

Reproducible ledger for the `production_assembly_distribution_network` learned-vs-own-best-heuristic
result on the env's `PRIMARY_REFERENCE_INSTANCE` `pirhooshyaran2021_serial_case3`. The `outputs/`
directory is gitignored, so the numbers below are embedded here as the tracked record.

## Honest status (read first)

- This env is FAITHFUL to the Pirhooshyaran & Snyder (2021, arXiv:2006.05608) network MDP
  (eq. 1-13, cost eq. 3), verified equation-by-equation in-crate, but it is
  **`literature_verified = false`**: there is NO published optimum for THIS network env.
- The ONLY literature-verified number for this family is the **single-node newsvendor cost
  127.11** (Table 1, mu=100, sigma=10, h=10, p=30, L=1, T=2), which the env's exact DP reproduces
  to within <1% (env DP at OUL=107 -> ~127.10). That verifies the env DYNAMICS, not the case3
  result. (Verified by `verification/tests.rs::single_node_newsvendor_cost_reproduced_by_exact_env_dp`.)
- The serial textbook optimum 47.65 is **structurally UNREACHABLE** by this env (echelon-level OULs
  applied to eq. 5's LOCAL raw-material position, which excludes finished goods). It is NOT a target
  here; its verified home is the sibling `multi_echelon/serial` family.
- The case3 comparator is therefore the env's OWN best grid-searched pairwise base-stock — a
  RESEARCH baseline, NOT a published optimum. The headline is "learned beats the env's own best
  heuristic", explicitly NOT "beats a literature benchmark".

## Instance (verbatim from `literature/references.rs` PRIMARY_REFERENCE_INSTANCE)

| field | value |
|---|---|
| name | `pirhooshyaran2021_serial_case3` |
| topology | 3-node serial 0 -> 1 -> 2; node 0 the only source |
| external demand | N(5,1) rounded/clipped at downstream node 2 only |
| horizon T | 10 periods |
| lead times | external->0 = 2, edge 0->1 = 1, edge 1->2 = 1 |
| local holding (up->down) | [2, 4, 7] |
| backorder cost | [0, 0, 37.12] (node 2 only) |
| supply relations (order) | relation 0 = edge(0->1), relation 1 = edge(1->2), relation 2 = external->node 0 |
| initial finished | [10, 5, 5]; pipelines [[0],[0],[0,0]]; all else 0 |
| objective | undiscounted average per-period cost |

## Evaluation protocol (like-for-like / paired CRN)

- Disjoint demand-path blocks: search seed 500_000, held-out seed 900_000. Demand only at node 2,
  N(5,1) rounded/clipped, T=10, undiscounted.
- Strongest in-env heuristic = best pairwise base-stock: grid-search per-relation OUL on the search
  block (`pairwise_base_stock` via `..._policy_rollout_from_paths`), re-score the argmin on the
  held-out block. This is the keep/discard gate.
- The SAME held-out block scores the learned soft-tree (via `..._soft_tree_rollout_from_paths`) and
  the pairwise base-stock — paired / variance-reduced.
- Policy = depth-2 (or 3) oblique linear-leaf soft tree over the `vector_quantity` per-supply-relation
  action (action_dim = 3), warm-started at the steady-state flow rate (order ~5/relation/period).

## Headline (full budget, refreshed 2026-06-04, install commit 2bb8df8)

Budget `full`: CMA-ES popsize 24, generations 60, train_seed_batch 96 (paired CRN), grid `fine`,
search 256 paths, held-out **4000 paths**. Box cap RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4.

The env's own best pairwise base-stock gate is identical across all runs (deterministic grid search):
**OUL = [8, 7, 9], held-out 60.2399 / period** (search-block 60.7236). This is the env's own argmin,
NOT an optimum. (The carried analytical Clark-Scarf levels cost far more here — echelon levels are
the wrong local targets.)

| config | params | gen-0 (flow warm-start) | learned held-out +/- SEM | gate (best pairwise BS) | gap (cost) | gap % | winner |
|---|---|---|---|---|---|---|---|
| depth2 oblique linear, seed 123 | 465 | 70.851 +/- 0.610 | **57.250 +/- 0.216** | 60.240 (OUL [8,7,9]) | -2.990 | **-4.96%** | learned |
| depth2 oblique linear, seed 321 | 465 | 70.851 +/- 0.610 | **54.958 +/- 0.232** | 60.240 (OUL [8,7,9]) | -5.282 | **-8.77%** | learned |
| depth3 oblique linear, seed 123 | 961 | 70.851 +/- 0.610 | **57.849 +/- 0.246** | 60.240 (OUL [8,7,9]) | -2.391 | **-3.97%** | learned |

Every config beats the gate by **> 3.9%**, robustly outside the held-out SEM (~0.22-0.25, i.e. by
> 9 SEM on the closest config). The committed headline (learned 57.25 +/- 0.22, -4.96%) reproduces
exactly. Confirmed across 2 CMA seeds and 2 tree depths.

## Why the learned policy beats the heuristic

The pairwise base-stock policy uses LOCAL raw-position feedback only
(`raw_inventory_by_relation - total_current_demand + in_transit + predecessor_backlog`, which
EXCLUDES finished goods). The learned linear-leaf direct-quantity policy can additionally read
finished inventory, internal/external backlog, and inbound pipeline per node, and switch order
behavior by inventory regime via oblique splits — a strictly richer control class on the SAME action
relations. Constant-leaf direct-quantity trees stay at the flow regime (~70/period) and lose,
confirming the leaf class (not optimizer budget) is the lever. This is the same "action design /
leaf class, not capacity, is the lever" thesis as the OWMR `direct_orders` -> structured and the
multi_echelon grid -> `direct_level` flips, here on a faithful-but-non-literature-verified network MDP.

## Reproduce

```
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py \
    --description "full refresh depth2 seed123" --budget full --depth 2 --seed 123
# repeat with --seed 321 and with --depth 3 --seed 123 for the robustness rows.
```

Smoke (validate-only, not a decision budget; heuristic wins at tiny budget by design):
`--budget smoke`. Each full run is ~16-21 s on 4 rayon cores on the shared box.

## Path-B feasibility note (recovering a published serial/network cost — investigation only)

See `PAPER_SECTION_DRAFT/README.md` for the full assessment. Summary: feasible-but-nontrivial, env dynamics
need NOT change. The lever is the policy/position definition, not the dynamics. The env already ships
`serial_echelon_simulation.rs::echelon_base_stock_requests` (orders to the ECHELON inventory
position). Reproducing the paper's 47.65 requires recovering Pirhooshyaran's exact OUL ->
inventory-position simulation protocol (warm-start and position convention), then optimizing the
LOCAL/echelon base-stock levels for THIS env rather than feeding carried echelon OULs to the local
policy. Until an env simulation re-derives a published cost within ~2%, the env stays
`literature_verified = false` and this result remains a research comparison.
