# Literature

Current literature anchor for `joint_replenishment`:

- Vanvuchelen, Gijsbrechts & Boute (2020), "Use of Proximal Policy Optimization for the Joint
  Replenishment Problem", Computers in Industry 119, 103239.
  Author copy: https://lirias.kuleuven.be/retrieve/badd4d5b-5bfc-44e4-84f1-b98fd113143d

Repo interpretation:

- the first carried slice is a small-scale multi-item setting with a shared full-truckload
  replenishment cost
- the Vanvuchelen action constraint is enforced as `sum_i q_i = M V`, so every nonzero shipment
  must use an exact integer number of full trucks
- benchmark policies and reduced exact verification are defined against that interpretation

Use `literature/references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE` (= setting 5, the family Figure 3/4 visualise)
- `VANVUCHELEN_2020_FIGURE3_ANCHOR` (the one exact executable anchor)
- `VERIFICATION_PROBLEM_INSTANCE` (the reduced finite-horizon self-consistency comparator)
- the carried small-scale settings and benchmark-policy names

Status:

- the Vanvuchelen Table 2 setting definitions are public and carried here verbatim (16 settings,
  V=6, K=75, k2=10, k1 in {10,40}, h/b permutations, demand U[0,5]/U[0,3] and U[0,6]/U[0,2])
- the paper reports per-setting optimality gaps only as a figure (Figure 2: heuristics 4-25% above
  optimal), so no full per-setting absolute-cost table is carried for assertions
- HOWEVER the paper states one exact, executable result in prose (Section 6.2, around Figure 3,
  setting 5): the optimal policy in state `(I1,I2)=(5,0)` orders `q=(0,6)` (one FTL to shipper 2),
  while both heuristics order `q=(2,4)`. An independent infinite-horizon value iteration over the
  repo env cost (Eq. 2) and balance (Eq. 4) reproduces the optimal `q=(0,6)`. This is a genuine
  literature anchor; the environment is therefore literature-verified against it.
- caveat: the repo MOQ/DYN-OUT are repo variant implementations of `(Q,S|T)` / Kiesmueller DYN-OUT
  fit to the same cost structure; their exact allocation in `(5,0)` need not equal the paper's
  `(2,4)`, so the heuristic action is carried for context, not as a repo assertion.

Citation correction (2026-05): an earlier version cited "Computers in Industry 122, 103300" with a
`merit.url.edu` URL; the correct reference is Computers in Industry 119, 103239 (KU Leuven lirias
author copy). Fixed in `references.rs`.
