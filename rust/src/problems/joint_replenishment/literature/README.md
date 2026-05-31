# Literature

Current literature anchor for `joint_replenishment`:

- Vanvuchelen, Gijsbrechts & Boute (2020), "Use of Proximal Policy Optimization for the Joint
  Replenishment Problem", Computers in Industry 119, 103239.
  DOI: https://doi.org/10.1016/j.compind.2020.103239
  Open author copy: https://lirias.kuleuven.be/retrieve/badd4d5b-5bfc-44e4-84f1-b98fd113143d
  (citation verified 2026-05 against Crossref, ScienceDirect PII S0166361519308218, and the author PDF
  metadata: authors and title exact, journal Computers in Industry, vol 119, article 103239, 2020.)

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

Status (audited 2026-05 against the source PDF; honest scope below):

CITATION: verified correct. Authors (Nathalie Vanvuchelen, Joren Gijsbrechts, Robert Boute), title,
journal (Computers in Industry), volume 119, article 103239, year 2020, and DOI
10.1016/j.compind.2020.103239 all match Crossref, ScienceDirect (PII S0166361519308218) and the author
PDF. (An earlier version of this file cited the wrong volume/article and a bogus URL; that is fixed.)

MODEL FIDELITY: literature-verified. The env equations match the paper exactly, confirmed in the PDF:
action constraint `sum_i q_i = M_t V` (Eq. 1), cost `c_t = sum_i[h_i[I]+ + b_i[-I]+ + k_i 1{q_i>0}] +
M_t K` (Eq. 2), state = previous-period end inventories (Eq. 3), balance `I_t = I_{t-1}+q-d` (Eq. 4),
order-before-demand with zero lead time and risk period one.

SETTING DEFINITIONS: literature-verified (carried verbatim). All 16 Table 2 settings confirmed against
the paper (V=6, K=75, k2=10, k1 in {10,40}, h/b in {1,5}/{19,95} permutations, demand U[0,5]/U[0,3] and
U[0,6]/U[0,2]). Every parameter in `SMALL_SCALE_SETTINGS` matches its Table 2 row.

PUBLISHED NUMBERS: the paper reports per-setting optimality gaps ONLY as a figure (Figure 2: heuristics
4-25% above optimal), so there is NO per-setting absolute-cost table to assert. The single exact,
quotable result is an OPTIMAL ACTION, not a cost: Section 6.2 (around Figure 3, setting 5) states
verbatim that in state `(I1,I2)=(5,0)` the optimal policy orders `q=(0,6)` (one FTL to shipper 2), the
PPO policy matches it, and both `(Q,S|T)` and DYN-OUT order `q=(2,4)`. This action is carried as
`VANVUCHELEN_2020_FIGURE3_ANCHOR` (state, action, gamma=0.99 all confirmed against the PDF).

REPRODUCTION SCOPE (be precise):
- In-crate (`verification/tests.rs`): asserts (a) the carried anchor's SHAPE and (b) the env's
  one-period cost at the STORED optimal action `q=(0,6)` for demand `(2,4)`, = 90 (Eq. 2/4). These
  confirm env cost-accounting fidelity; they do NOT re-derive that `q=(0,6)` is the optimum.
- External only: the claim that `q=(0,6)` is the infinite-horizon (gamma=0.99) value-iteration OPTIMUM
  is reproduced by `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`, which lives outside
  this crate and is not part of `cargo test`. Treat the optimality reproduction as faithful-but-external,
  not as an in-crate literature assertion.

Therefore the accurate overall status is PARTIAL: model + Table-2 settings are literature-verified and a
published one-period cost identity is reproduced in-crate, but the headline "optimal action `q=(0,6)`"
reproduction is external/not-in-crate, and the only published benchmark scalar (optimality gaps) is a
figure that cannot be reproduced to a number.

caveat: the repo MOQ/DYN-OUT are repo variant implementations of `(Q,S|T)` (Cachon, 2001) / DYN-OUT
(Kiesmueller, 2009) fit to the same cost structure; their exact allocation in `(5,0)` need not equal the
paper's `(2,4)`, so the heuristic action is carried for context, not as a repo assertion.

Citation correction (2026-05): an earlier version cited "Computers in Industry 122, 103300" with a
`merit.url.edu` URL; the correct reference is Computers in Industry 119, 103239 (KU Leuven lirias
author copy). Fixed in `references.rs`.
