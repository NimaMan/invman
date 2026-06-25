# Fixed-Cost Autoresearch

Canonical fixed-order-cost benchmark:

- lost sales with `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- demand `~ Poisson(5)`
- holding cost `h=1`

Autoresearch workflow:

1. Broad fixed-cost tree screening identified depth-1 linear-leaf trees as the first strong alternative.
2. A focused depth-1 linear screen compared temperatures `{0.1, 0.25, 0.5}` and CMA sigma `{2, 5}`.
3. The best screening candidate was promoted to the full budget and then re-evaluated on a `1,000,000`-period horizon.

Focused screening winner:

- run summary:
  `invman/outputs/autoresearch/fixed_cost_d1_linear_screen/fixed_order_tree_search_screening.json`
- best screened candidate:
  - oblique split
  - depth `1`
  - linear leaves
  - temperature `0.25`
  - sigma `5.0`
  - screening mean cost: `8.77891`

Promoted full-budget result:

- result file:
  `invman/outputs/autoresearch/fixed_cost_promoted_d1_full/results/fixed_cost_promoted_d1_full_full_soft_tree_oblique_linear_d1_t0p25_s5p0.json`
- model dir:
  `invman/outputs/autoresearch/fixed_cost_promoted_d1_full/models/fixed_cost_promoted_d1_full_full_soft_tree_oblique_linear_d1_t0p25_s5p0_15_2000`
- training protocol:
  - CMA-ES
  - `2000` iterations
  - population `10`
  - rollout horizon `2000`
  - `same_seed` enabled within each ES batch
- `50k` evaluation:
  - learned policy: `8.77528`
  - `s,S`: `9.54290`
  - `s,nQ`: `9.21239`
  - modified `s,S,q`: `9.20292`

Long-horizon reevaluation:

- reevaluation file:
  `invman/outputs/autoresearch/fixed_cost_promoted_d1_full/results/fixed_cost_promoted_d1_full_full_soft_tree_oblique_linear_d1_t0p25_s5p0_eval1m.json`
- `1M` evaluation with warm-up:
  - learned policy: `8.76576`

Comparison to the previous fixed-cost tree benchmark:

- earlier best transferred tree, `1M` eval: `8.81009`
- new autoresearch tree, `1M` eval: `8.76576`
- absolute improvement: `0.04433`
- relative improvement: about `0.50%`

Comparison to the best heuristic baseline on the canonical instance:

- best heuristic on `1M` eval, modified `s,S,q`: `9.16537`
- new autoresearch tree, `1M` eval: `8.76576`
- relative improvement: about `4.36%`

Current conclusion:

- the best fixed-cost learned policy is now a shallower tree than the vanilla winner
- vanilla best tree: oblique depth-2 with linear leaves
- fixed-cost best tree: oblique depth-1 with linear leaves
- fixed ordering cost changes the best search space enough that autoresearch over policy structure is worthwhile
