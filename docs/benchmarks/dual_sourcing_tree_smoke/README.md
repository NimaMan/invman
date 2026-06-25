# Dual-Sourcing Tree Smoke Result

Primary instance:

- regular lead time `l_r = 4`
- expedited lead time `l_e = 0`
- expedited cost `c_e = 110`
- regular cost `c_r = 100`
- holding cost `h = 5`
- backlog cost `b = 495`
- demand uniform on `{0,1,2,3,4}`

Learned policy run:

- result file: `invman/outputs/results/dual_sourcing_soft_tree_l4_ce110_smoke.json`
- policy family: soft tree
- split type: oblique
- depth: `2`
- leaf type: linear
- training budget: `120` CMA-ES iterations, population `8`
- evaluation summary:
  - learned tree: `249.84`

Heuristic benchmark on the same run:

- capped dual-index: `220.73`
- dual-index: `221.54`
- tailored base-surge: `221.72`
- single-index: `225.87`

Interpretation:

- the dual-sourcing package, heuristic search, and Rust rollout path are working;
- under the first smoke budget, the learned tree is not yet competitive with the benchmark heuristics.
