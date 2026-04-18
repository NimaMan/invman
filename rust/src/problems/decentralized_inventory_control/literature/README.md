# Literature

Current literature anchors:

- Sterman (1989), original Beer Game benchmark
- Caner et al. (2014), exact board-game reconstruction and public verification code
- Oroojlooyjadid et al. (2021), DQN Beer Game benchmark row
- Mousa et al. (2024), broader decentralized local-information inventory-control context

Reference split:

- [PRIMARY_REFERENCE_INSTANCE](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs) is the classic four-stage Beer Game with the canonical 36-week demand path `4,4,4,4,8,...,8`
- the classic Sterman benchmark row is `[46, 50, 54, 54]`, total `204`
- the Oroojlooy Sterman row is `[10.81, 10.76, 10.96, 12.60]`, total `45.13`

Verification status:

- classic Sterman/Caner benchmark: literature-verified through the Rust port of the public Caner verification code
- Oroojlooy 2021 benchmark row: published row only, not yet verified by the repo implementation

Use [references.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs) as the source of truth for:

- literature sources
- primary benchmark instance
- reduced repo verification instance
- published benchmark rows
