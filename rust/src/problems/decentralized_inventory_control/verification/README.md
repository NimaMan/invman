# Verification

Verification lives in this folder.

## Honest status (audited 2026-05-31)

There are TWO distinct simulators in this problem, and they are NOT the same model:

1. `verification/classic_board_game.rs` — a standalone, closed-form port of the public
   Edali & Yasarcan (2014) R code (JASSS 17(4) 2). It hardcodes the optimized
   anchor-and-adjust ordering policy and the board-game pipeline bookkeeping
   (two-period shipment delay split into `iti1`/`iti2`/`wipi`, supply line that
   includes the upstream agent's backlog). With the published parameters
   theta=0, sat=1, wsl=1, S'=[28,28,28,20], h=0.5, p=1.0 it reproduces the classic
   Sterman benchmark exactly: per-stage `[46, 50, 54, 54]`, total `204`.
   This is literature-verified.

2. `env.rs` — the reusable decentralized serial-control environment that the
   `base_stock` / `sterman_anchor_adjust` heuristics, the exact finite-horizon DP,
   and the learned soft-tree rollout all run on. It is a different (also valid)
   decentralized serial inventory MDP: receive shipments -> fill demand+backlog ->
   ship downstream -> place orders, with cost on ending on-hand and backlog. Its
   `on_order` (env.rs:150-164) omits the upstream agent's backlog and uses a
   fixed-length shipment-pipeline list rather than the board-game `iti1/iti2/wipi`
   split.

### The gap (root cause)

The `S'=[28,28,28,20]` anchor-and-adjust targets were tuned to the closed-form
board-game bookkeeping. They do NOT transfer to `env.rs`. Running the repo's
`sterman_anchor_adjust` heuristic with those exact parameters on
`PRIMARY_REFERENCE_INSTANCE` through `env.rs` (binding
`decentralized_inventory_control_policy_rollout_from_paths`, discount 1.0,
36-week path 4,4,4,4,8,...,8) yields **378**, not 204. The two models reach
different steady states: the closed-form model starves to on-hand 0 with a long
supply line (orders 8, on-order ~28/24), whereas `env.rs` settles at on-hand 4 at
every stage and never backlogs. The best simple base-stock on `env.rs` (S=24)
reaches **278**, still well above the closed-form 204 — confirming the gap is
structural (a different MDP), not a parameter-transfer artifact.

### Therefore

- The reusable `env.rs` environment is **NOT literature-verified**: it carries only
  one published anchor (the Sterman 204), and that anchor is reproduced only by the
  disconnected closed-form simulator, not by `env.rs` itself.
- `VERIFICATION_PROBLEM_INSTANCE` is honestly self-labeled
  `literature_verified: false` / `repo_exact_solver_not_verified_against_literature`.
  It only checks repo-internal consistency.

## Current scope

- [classic_board_game.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/classic_board_game.rs)
  - exact Rust port of the public Edali & Yasarcan (2014) Beer-Game R code
  - reproduces the classic Sterman benchmark exactly: `[46, 50, 54, 54]`, total `204`
  - closed-form, hardcoded anchor-and-adjust policy; does NOT exercise `env.rs`
- [tests.rs](/home/nima/code/ml/invman/rust/src/problems/decentralized_inventory_control/verification/tests.rs)
  - reference-shape checks
  - local policy-state layout checks
  - worked-transition accounting checks for the reusable serial environment (`env.rs`)
  - reduced finite-horizon DP checks on the repo-native verifier (exact DP dominates heuristics)

## Important distinctions

- the **closed-form board-game** benchmark is literature-verified (204)
- the **reusable env.rs** is NOT literature-verified — it reproduces 378/278, not 204,
  under the published parameters
- the reduced finite-horizon DP instance is repo-native and only verifies internal
  implementation consistency
- repo-native worked-transition expected values live in verification tests, not in the
  literature references

## What would make env.rs literature-verified (deferred, see top-level README next steps)

Either (a) re-derive `S'` and the supply-line definition so that env.rs's anchor-and-adjust
reproduces 204 on the canonical path (requires matching the board-game `iti1/iti2/wipi`
two-period shipment split and including the upstream backlog in the supply line — a
non-localized env.rs change that must be validated against a Rust rebuild), or
(b) adopt a different published decentralized-serial anchor whose convention matches
env.rs's order-after-demand bookkeeping (e.g. a Clark-Scarf serial base-stock instance
with a published optimal cost), and carry that as the verification target.
