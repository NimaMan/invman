# Verification

Verification lives in this folder.

Current scope:

- [classic_board_game.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/classic_board_game.rs)
  - exact Rust port of the public Caner Beer-Game verification code
  - reproduces the classic Sterman benchmark exactly: `[46, 50, 54, 54]`, total `204`
- [tests.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/tests.rs)
  - reference-shape checks
  - local policy-state layout checks
  - worked-transition accounting checks for the reusable serial environment
  - reduced finite-horizon DP checks on the repo-native verifier

Important distinction:

- the classic board-game benchmark is literature-verified
- the reduced finite-horizon DP instance is repo-native and only verifies internal implementation consistency
