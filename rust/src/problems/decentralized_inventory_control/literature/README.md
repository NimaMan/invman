# Literature

Current literature anchors:

- Sterman (1989), original Beer Game benchmark (Management Science 35(3), 321-339)
- Edali & Yasarcan (2014), exact board-game reconstruction and public R verification code
  (JASSS 17(4) 2, DOI 10.18564/jasss.2555). NOTE: earlier text in this folder mis-attributed
  this paper to "Caner et al."; the correct authors are Edali & Yasarcan. The internal
  constant is still named `CANER_2014_REFERENCE` for stability, but its `source` field now
  carries the correct citation.
- Oroojlooyjadid et al. (2021), RL Beer Game background paper (MSOM 24(1))
- Mousa et al. (2024), broader decentralized local-information inventory-control context

Reference split:

- [PRIMARY_REFERENCE_INSTANCE](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs)
  is the classic four-stage Beer Game with the canonical 36-week demand path `4,4,4,4,8,...,8`
- the classic Sterman benchmark row is `[46, 50, 54, 54]`, total `204`

## Verification status (audited 2026-05-31)

- classic Sterman / Edali-Yasarcan benchmark (`204`): literature-verified, but ONLY through
  the standalone closed-form simulator `verification/classic_board_game.rs` (an exact port of
  the public R code, with the optimized anchor-and-adjust policy hardcoded).
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
