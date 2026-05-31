# Literature

Current literature anchors (all four citations independently re-verified against Crossref /
publisher DOIs on 2026-05-31; see per-line URLs below):

- Sterman, J. D. (1989), "Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic
  Decision Making Experiment", Management Science 35(3):321-339, DOI 10.1287/mnsc.35.3.321.
  Verified via Crossref (https://doi.org/10.1287/mnsc.35.3.321). Sole author John D. Sterman (MIT
  Sloan). This is the original Beer Game / anchor-and-adjust paper.
- Edali, M. & Yasarcan, H. (2014), "A Mathematical Model of the Beer Game", Journal of Artificial
  Societies and Social Simulation (JASSS) 17(4):2, DOI 10.18564/jasss.2555. Verified via Crossref
  and the DOI redirect to jasss.org/17/4/2.html. Exact board-game reconstruction with public R
  verification code. NOTE: earlier text in this folder mis-attributed this paper to "Caner et al.";
  the correct authors are Mert Edali & Hakan Yasarcan. The internal constant is still named
  `CANER_2014_REFERENCE` for stability, but its `source` field now carries the correct citation.
- Oroojlooyjadid, A., Nazari, M., Snyder, L. V. & Takac, M., "A Deep Q-Network for the Beer Game:
  Deep Reinforcement Learning for Inventory Optimization", Manufacturing & Service Operations
  Management 24(1):285-304, DOI 10.1287/msom.2020.0939. Verified via Crossref. The bound-issue year
  is 2022; it was posted Articles-in-Advance in 2021. The repo constant is named
  `OROOJLOYJADID_2021_REFERENCE` (online-first year); the `source` string now states both years.
  RL Beer Game background paper.
- Mousa, M., van de Berg, D., Kotecha, N., del Rio-Chanona, E. A. & Mowbray, M. (2024), "An analysis
  of multi-agent reinforcement learning for decentralized inventory control systems", Computers &
  Chemical Engineering 188:108783, DOI 10.1016/j.compchemeng.2024.108783. Verified via Crossref.
  Broader decentralized local-information inventory-control context; background only.

Reference split:

- [PRIMARY_REFERENCE_INSTANCE](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs)
  is the classic four-stage Beer Game with the canonical 36-week demand path `4,4,4,4,8,...,8`
- the classic Sterman benchmark row is `[46, 50, 54, 54]`, total `204`

## Verification status (audited 2026-05-31)

- classic Sterman / Edali-Yasarcan benchmark (`204`): reproduced by a solver, but ONLY through
  the standalone closed-form simulator `verification/classic_board_game.rs` (an exact port of
  the public R code, with the optimized anchor-and-adjust policy hardcoded). Citation caveat
  from the 2026-05-31 audit: Edali & Yasarcan (2014) sec 5 explicitly states it "obtained the
  exact same benchmark cost values reported by Sterman" under theta=0, sat=1, wsl=1,
  S'=[28,28,28,20], h=0.5, p=1.0 (exactly the repo parameters), but the specific per-stage split
  `[46,50,54,54]`/total `204` is NOT quoted in the freely-accessible text of either paper (the
  open Sterman 1989 copy is an image-only scan). The `204` here is therefore the value the repo's
  R-port emits under the published parameters, consistent with the paper's claim, rather than a
  figure transcribed from a quotable published table. Treat it as a solver-reproduced anchor for
  the closed-form simulator, pending a paywalled-source confirmation of the exact split (see the
  top-level audit blockers).
- the reusable `env.rs` transition (the one the heuristics, exact DP, and learned soft-tree
  actually run on) is **NOT literature-verified**. It is a different, also-valid decentralized
  serial MDP whose pipeline/supply-line bookkeeping differs from the board game, so the
  published `S'=[28,28,28,20]` parameters do not transfer: anchor-and-adjust through `env.rs`
  costs `378` and the best simple base-stock (`S=24`) costs `278` on the canonical 36-week
  path, vs the closed-form `204`. See [verification/README.md](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/README.md)
  for the measured numbers and root cause.
- Oroojlooy 2021 paper: background only, not carried as a repo benchmark row.

Use [references.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs) as the source of truth for:

- literature sources
- primary benchmark instance (closed-form anchor)
- reduced repo verification instance (repo-internal only, not literature-verified)
- published benchmark rows that the repo can actually defend as executable anchors

Oroojlooy audit result:

- the paper's `45.13` Sterman row belongs to a 100-period uniform-demand benchmark, not the
  classic 36-week board-game slice
- the public paper appendix, the reported benchmark description, and the released GitHub code
  do not line up tightly enough to recover that row to repo verification accuracy
- so the repo does not carry the `45.13` row as a benchmark assertion
