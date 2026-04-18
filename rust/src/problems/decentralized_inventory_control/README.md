# decentralized_inventory_control

Rust-first home for decentralized serial inventory control with Beer-Game-style local observations.

Formulation carried here:

- four-stage serial chain in the classic literature slice: retailer, wholesaler, distributor, factory
- each stage observes only local inventory, backlog, pipeline, and incoming-order information
- orders move upstream with information delay, shipments move downstream with physical delay
- per-period cost is linear holding plus backlog cost

This package currently carries two literature slices:

- classic board-game benchmark from Sterman (1989), reconstructed exactly by Caner et al. (2014)
- later RL background paper from Oroojlooyjadid et al. (2021)

Current status:

- literature-verified: yes for the classic Sterman/Caner benchmark
- not carried as a benchmark row: the Oroojlooyjadid 2021 `45.13` Sterman number could not be reproduced tightly enough from the public paper plus released code
- repo-exact verified: yes on the reduced finite-horizon verifier

Folder roles:

- [literature/README.md](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/README.md)
- [verification/README.md](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/README.md)
- [experiments/README.md](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/experiments/README.md)
- [practical/README.md](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/practical/README.md)

Code layout:

- root env / rollout / heuristics: reusable decentralized serial-control environment
- [references.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/literature/references.rs): literature rows and problem instances
- [classic_board_game.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/classic_board_game.rs): exact Rust port of the public Caner verification code
- [tests.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/tests.rs): package verification assertions
