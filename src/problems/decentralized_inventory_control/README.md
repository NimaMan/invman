# decentralized_inventory_control

Rust-first home for decentralized serial inventory control with Beer-Game-style local observations.

Formulation carried here:

- four-stage serial chain in the classic literature slice: retailer, wholesaler, distributor, factory
- each stage observes only local inventory, backlog, pipeline, and incoming-order information
- orders move upstream with information delay, shipments move downstream with physical delay
- per-period cost is linear holding plus backlog cost

This package currently carries two literature slices:

- classic board-game benchmark from Sterman (1989), "Modeling Managerial Behavior: Misperceptions
  of Feedback in a Dynamic Decision Making Experiment", Management Science 35(3):321-339
  (DOI 10.1287/mnsc.35.3.321), reconstructed exactly by Edali & Yasarcan (2014)
  (JASSS 17(4):2, DOI 10.18564/jasss.2555; earlier text mis-attributed this to "Caner et al.")
- later RL background paper from Oroojlooyjadid, Nazari, Snyder & Takac, "A Deep Q-Network for the
  Beer Game", MSOM 24(1):285-304, DOI 10.1287/msom.2020.0939 (issue year 2022; online-first 2021)

All four citations were independently re-verified against Crossref / publisher DOIs on 2026-05-31.

Current status (audited 2026-05-31 — corrected from an earlier overstated claim):

- The closed-form board-game simulator `verification/classic_board_game.rs` is
  literature-verified: it reproduces the Sterman benchmark `[46, 50, 54, 54]`, total `204`
  exactly (confirmed via the installed binding
  `decentralized_inventory_control_classic_sterman_literature_summary`).
- The reusable `env.rs` environment — the one the heuristics, the exact finite-horizon DP, and
  the learned soft-tree actually run on — is **NOT literature-verified**. It is a different,
  also-valid decentralized serial inventory MDP. The board-game `S'=[28,28,28,20]` anchor-and-adjust
  parameters do not transfer to it: through `env.rs` the same policy costs `378`, and the best
  simple base-stock (`S=24`) costs `278` on the canonical 36-week path, vs the closed-form `204`.
  The gap is structural (different pipeline/supply-line bookkeeping), not a tuning artifact.
- not carried as a benchmark row: the Oroojlooyjadid 2021 `45.13` Sterman number could not be
  reproduced tightly enough from the public paper plus released code.
- repo-exact verified: yes on the reduced finite-horizon verifier (`VERIFICATION_PROBLEM_INSTANCE`
  is honestly self-labeled `literature_verified: false`; the exact DP dominating the heuristics is
  a repo-internal consistency check only).

What was investigated and fixed in this pass:

- Fixed the false author attribution (Caner et al. -> Edali & Yasarcan, 2014) in `references.rs`.
- Root-caused and documented why `env.rs` does not reproduce the published `204`: see
  [verification/README.md](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/verification/README.md).
- Corrected the previously overstated "literature-verified: yes" status to distinguish the
  verified closed-form simulator from the unverified reusable environment.

Remaining steps to make `env.rs` literature-verified (deferred — require a Rust rebuild and
therefore not done here): either re-derive the supply-line definition and `S'` so env.rs's
anchor-and-adjust reproduces `204` (match the board-game two-period shipment split and include
the upstream backlog in the supply line), or adopt a published decentralized-serial anchor whose
order-after-demand convention matches env.rs (e.g. a Clark-Scarf serial base-stock instance with
a known optimal cost) and carry that as the verification target. A ready-to-run measurement
script is at `scripts/decentralized_inventory_control/measure_env_vs_closedform.py`.

Folder roles:

- [literature/README.md](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/literature/README.md)
- [verification/README.md](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/verification/README.md)
- [experiments/README.md](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/experiments/README.md)
- [practical/README.md](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/practical/README.md)

Code layout:

- root env / rollout / heuristics: reusable decentralized serial-control environment
- [references.rs](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/literature/references.rs): literature rows and problem instances
- [classic_board_game.rs](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/verification/classic_board_game.rs): exact Rust port of the public Edali & Yasarcan (2014) verification R code (closed-form board game; not the reusable env.rs)
- [tests.rs](/home/nima/code/ml/invman/src/problems/decentralized_inventory_control/verification/tests.rs): package verification assertions
