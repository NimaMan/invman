# Fixed-Cost Tree Transfer

Canonical fixed-order-cost benchmark:

- lost sales with `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- demand `~ Poisson(5)`
- holding cost `h=1`

Original learned transfer policy:

- architecture: oblique depth-2 soft tree with linear leaves
- training protocol: CMA-ES, `2000` iterations, population `10`, rollout horizon `2000`
- result file:
  `invman/outputs/results/soft_tree_oblique_linear_fixed_cost_l4_p4_k5_pois5_2k_pop10.json`

Evaluation results:

- learned soft tree, `50k` eval: `8.81895`
- learned soft tree, `1M` eval with `20%` warm-up: `8.81009`

Heuristic comparison on `1M` evaluation:

- `s,S`: `9.44401`
- `s,nQ`: `9.21664`
- modified `s,S,q`: `9.16537`

So the original transferred tree improves on the best heuristic by about `3.9%` on the canonical
fixed-cost instance.

Autoresearch follow-up:

- documented in `fixed_cost_autoresearch/README.md`
- improved architecture:
  - oblique depth-1 soft tree
  - linear leaves
- improved `1M` result: `8.76576`

So the current best fixed-cost policy is no longer the transferred depth-2 tree; it is the
autoresearch-refined depth-1 linear-leaf tree.

Sanity check:

- a second heuristic search with a longer search horizon (`10000`) did not overturn the result
- best candidates from that re-search were:
  - `s,S = (21, 26)` -> `9.45251`
  - `s,nQ = (22, 8)` -> `9.18528`
  - modified `s,S,q = (21, 28, 8)` -> `9.18943`

So the earlier heuristic benchmark remains the stronger comparator, and the tree still wins
comfortably.
