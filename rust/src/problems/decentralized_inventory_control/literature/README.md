# Literature

Current literature anchors:

- Sterman (1989), original Beer Game benchmark
- Caner et al. (2014), exact board-game reconstruction and public verification code
- Oroojlooyjadid et al. (2021), RL Beer Game background paper
- Mousa et al. (2024), broader decentralized local-information inventory-control context

Reference split:

- [PRIMARY_REFERENCE_INSTANCE](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs) is the classic four-stage Beer Game with the canonical 36-week demand path `4,4,4,4,8,...,8`
- the classic Sterman benchmark row is `[46, 50, 54, 54]`, total `204`

Verification status:

- classic Sterman/Caner benchmark: literature-verified through the Rust port of the public Caner verification code
- Oroojlooy 2021 paper: background only, not carried as a repo benchmark row

Use [references.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs) as the source of truth for:

- literature sources
- primary benchmark instance
- reduced repo verification instance
- published benchmark rows that the repo can actually defend as executable anchors

Oroojlooy audit result:

- the paper’s `45.13` Sterman row belongs to a 100-period uniform-demand benchmark, not the classic 36-week board-game slice
- the public paper appendix, the reported benchmark description, and the released GitHub code do not line up tightly enough to recover that row to repo verification accuracy
- so the repo does not carry the `45.13` row as a benchmark assertion
