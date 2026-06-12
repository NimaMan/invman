# decentralized_inventory_control — benchmark card

**One-line MDP:** four-stage serial chain (retailer → wholesaler → distributor → factory), each
stage sees only its own local inventory / backlog / pipelines / incoming orders; **action** = the
order each stage places upstream; **one-period cost** = sum over stages of linear holding (on local
on-hand) plus linear backlog; **objective** = minimize total (undiscounted, 36-week horizon)
holding + backlog cost across the chain.

**Status:** `verified_rerun` for the closed-form board-game port (Sterman/Edali-Yasarcan 204
reproduced EXACTLY by re-run); `faithful_unverified` for the reusable `env.rs` MDP that the
heuristics / exact DP / learned soft-tree actually run on (it does NOT reproduce 204 — under the
identical published parameters it yields 378, best simple base-stock 278). The two coexist: the
closed-form simulator is a disconnected literature anchor, not the trainable environment.
**Paper:** referenced only as background in §"Related work" of
`learning_inventory_control_policies_es.tex` (the beer-game DQN citation, Oroojlooyjadid et al.);
this system has **no trained results section** in the paper. The single positive `env.rs` property
(Clark-Scarf constant-demand serial optimum = 0.0) is adjacent to the serial Clark-Scarf system of
§"Serial multi-echelon (Clark--Scarf)" (`sec:serial`), which is a separate problem family.

## Problem formulation
Decentralized serial inventory control with Beer-Game-style local information.

- **Timing (per period).** Customer demand realizes at the retailer (exogenous; the retailer order
  pipeline must be empty). Each stage receives its due inbound shipment (from the downstream end of
  its shipment pipeline) and the due inbound order from the stage below; orders move upstream with an
  information delay, shipments move downstream with a physical delay (canonical 4-stage instance:
  shipment lead `[2,2,2,2]`, order lead `[0,1,1,1]`). Each stage ships as much as it can toward its
  customer (filling backlog first), then places its own order upstream.
- **State** (`DecentralizedInventoryControlState`): `period`, per-stage `on_hand_inventory`,
  `backlog`, `shipment_pipelines`, `order_pipelines`, `last_received_shipments`,
  `last_received_orders`, `forecast_orders`, `last_actions`. Each stage observes only its own
  components — this is the decentralized / local-information structure. Requires ≥2 stages;
  all non-retailer order pipelines and all shipment pipelines must have strictly positive lead time.
- **Action.** Order quantity placed upstream by each stage (the retailer's "upstream" is the
  wholesaler; the factory orders from an external source).
- **Transition.** Shipment/order pipelines advance one step; inbound shipments add to on-hand,
  demand/incoming orders draw down on-hand and accumulate backlog when short.
- **One-period cost.** `period_cost = Σ_stages (holding[i]·ending_on_hand[i] + backlog[i]·ending_backlog[i])`;
  `reward = -period_cost`. Canonical costs: holding `0.5`, backlog `1.0` for every stage.
- **Objective.** Minimize total undiscounted cost over the 36-week horizon (discount `1.0` in the
  classic instance). The reduced verifier instance uses discount `0.99`.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `beer_game_classic_four_stage` (PRIMARY_REFERENCE_INSTANCE) | regime serial_backlog; stages 4; ship lead `[2,2,2,2]`, order lead `[0,1,1,1]`; deterministic step demand 4→8; horizon 36 weeks; holding 0.5 / backlog 1.0; Sterman anchor `S'=[28,28,28,20]` | demand path `4,4,4,4,8,…,8` (36 wk); on-hand `[12,12,12,12]`; smoothing θ=0; supply-line weight 1; adj. time 1; `published_sterman_benchmark Some(204)` | **absent (no `literature_verified` field).** Carries the published 204 anchor but `env.rs` does NOT reproduce it: anchor-and-adjust → 378, best base-stock → 278. Literature anchor for the *closed-form simulator only*, NOT a verification target for `env.rs`. |
| `STERMAN_1989_CLASSIC_BENCHMARK` (published row, closed-form only) | published_row sterman_anchor_adjust; per-agent `[46,50,54,54]`; total `204`; scope = closed-form board game only | policy `sterman_anchor_adjust`; `per_agent_mean_costs [46,50,54,54]`; `total_mean_cost 204.0` | **absent.** Notes warn explicitly that 204 is a property of the closed-form board-game bookkeeping only (it is the value the R-port *emits*, not a transcribed published table line); `env.rs` yields 378. |
| `VERIFICATION_PROBLEM_INSTANCE` (reduced exact-DP) | regime serial_backlog; stages 2; horizon 3 periods; discount 0.99; demand Bernoulli{0,1} p=0.5; max_order 4; base_stock 3; scope repo-internal only | targets `[4,4]`, adj. time `[1,1]`, supply-line weight `[1,1]` | **false** (`repo_exact_solver_not_verified_against_literature`) — honestly self-labeled in `references.rs`. |

## Baselines
- **Heuristics:** `base_stock` (installation base-stock; in the audit searched by a single shared
  level sweep `S ∈ {16,20,24,28}` on `env.rs`, best at S=24) and `sterman_anchor_adjust`
  (anchor-and-adjust with per-stage targets / adjustment-times / supply-line-weights).
- **Exact / optimal:** `finite_horizon_dp.rs` — bounded backward-induction DP over the reduced
  2-agent / 3-period `VERIFICATION_PROBLEM_INSTANCE`. **Test-only, NOT a Python binding.** It
  provably dominates both heuristics on that reduced instance (a repo-internal consistency check
  only). The classic 4-stage 36-week instance has **NO exact optimum carried.**
- **Published comparators (CONTEXT only):**
  - Sterman (1989) / Edali-Yasarcan (2014) anchor-and-adjust, per-stage `[46,50,54,54]`, total
    `204` (36-week classic Beer Game) — **closed-form board-game only**; 204 is the value the
    R-port emits, not a transcribed paper table.
  - Oroojlooyjadid et al. (2022) "A Deep Q-Network for the Beer Game" Sterman row `45.13` —
    **DELIBERATELY NOT carried** (different benchmark; the public paper's formula/timing and released
    code do not line up tightly enough to carry as an executable anchor). This is a cross-protocol
    DRL number and is never a "beat" target.

## Verification
- **Published number:** Sterman / Edali-Yasarcan anchor-and-adjust per-stage `[46,50,54,54]`,
  total `204` (36-week classic Beer Game).
- **Re-run reproduced:** closed-form `classic_board_game.rs` → per-stage `[46.0,50.0,54.0,54.0]`,
  total `204.0` (**EXACT**), via
  `python -c "import invman_rust; print(invman_rust.decentralized_inventory_control_classic_sterman_literature_summary())"`
  (or the full audit script `measure_env_vs_closedform.py`). **Verdict: `verified_rerun` for the
  closed-form port only.**
- **env.rs debt (`faithful_unverified`).** Under the SAME published parameters
  (`S'=[28,28,28,20]`, θ=0, sat=1, wsl=1, h=0.5, p=1.0, 36-week path), the reusable `env.rs`
  transition yields `sterman_anchor_adjust → 378.0` and best simple base-stock `S=24 → 278.0` — it
  does NOT reproduce 204. The gap is **structural** (different pipeline/supply-line bookkeeping),
  not a tuning artifact. `env.rs` is a different, also-valid decentralized serial MDP. The only
  positive published-flavored property `env.rs` does reproduce is the Clark-Scarf constant-demand
  serial optimum (4-stage → 0.0, 2-stage → 0.0).
- **Reduced exact DP:** `VERIFICATION_PROBLEM_INSTANCE` is honestly self-labeled
  `literature_verified: false`; the DP dominating the heuristics is a repo-internal consistency
  check, not a literature verification.

## Results (learned policy)
- **None carried.** The manifest `results` array is empty for this system. There is no
  seed-robust trained-policy row, and no "beats gate" claim — neither at-risk (single-seed /
  best-of-N) nor seed-robust. This is a benchmark/verification entry: a `verified_rerun` closed-form
  anchor plus a `faithful_unverified` trainable `env.rs`, with no learned-policy result to report.

## Reproduce
```bash
# Full audit: closed-form 204 vs env.rs 378 / best base-stock 278, plus Clark-Scarf 0.0 check
python /home/nima/code/ml/invman/scripts/decentralized_inventory_control/measure_env_vs_closedform.py

# Closed-form literature-verified anchor (per-stage [46,50,54,54], total 204)
python -c "import invman_rust; print(invman_rust.decentralized_inventory_control_classic_sterman_literature_summary())"

# env.rs base-stock rollout via the path-based binding (decentralized serial MDP)
python -c "import invman_rust as r; print(r.decentralized_inventory_control_policy_rollout_from_paths('base_stock',[32.0]*4,[0]*4,[0]*4,[[8,8],[8,8],[8,8],[8,8,8,8]],[[],[8,8],[8,8],[8,8]],[8]*4,[8]*4,[8.0]*4,[8]*4,[8]*100,[0.0]*4,[0.5]*4,[1.0]*4,1.0))"

# List available bindings for this problem
python -c "import invman_rust; print([n for n in dir(invman_rust) if 'decentralized' in n])"
```

## Pointers & caveats
- code: `src/problems/decentralized_inventory_control/env.rs` (reusable decentralized serial MDP),
  `rollout.rs`, `heuristics/`, `finite_horizon_dp.rs` (test-only reduced exact DP),
  `literature/references.rs` (instances + published rows),
  `verification/classic_board_game.rs` (exact Rust port of the public Edali & Yasarcan 2014 R code —
  the closed-form board game, NOT `env.rs`), `verification/tests.rs`, `bindings.rs`, `demand.rs`.
- scripts: `scripts/decentralized_inventory_control/measure_env_vs_closedform.py`.
- autoresearch: no `policy_search/programs/program_decentralized_inventory_control.md` exists (no autoresearch
  program for this system).
- **Caveat — two-MDP split.** The literature-verified `204` lives ONLY in the disconnected
  closed-form port; the trainable `env.rs` is `faithful_unverified` and yields 378/278 under the
  same params. Do not conflate them. To make `env.rs` literature-verified, either re-derive its
  supply-line definition / `S'` so anchor-and-adjust reproduces 204 (match the board-game two-period
  shipment split and include upstream backlog in the supply line), or adopt a published
  decentralized-serial anchor whose order-after-demand convention matches `env.rs` (e.g. a
  Clark-Scarf serial base-stock instance with a known optimum). Both require a Rust rebuild and were
  deferred.
- **Caveat — cross-protocol DRL.** Oroojlooyjadid et al. (2022) `45.13` is CONTEXT only and is
  deliberately not carried; never treat it as a "beat" target.
- **Caveat — citation history.** An earlier README mis-attributed the verification reconstruction to
  "Caner et al."; corrected to Edali & Yasarcan (2014) in `references.rs`. All four citations
  (Sterman 1989; Edali-Yasarcan 2014; Oroojlooyjadid et al. 2022) re-verified against Crossref on
  2026-05-31.
- **Existing README.md:** `src/problems/decentralized_inventory_control/README.md` is consistent
  with the manifest and ledger (it already states the closed-form-verified / env.rs-unverified
  split honestly); this card does not contradict or overwrite it.
