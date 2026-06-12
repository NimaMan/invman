# Benchmark card — `decentralized_inventory_control`

**Subfamily:** four-stage Beer-Game serial under local information (closed-form board-game port + reusable env.rs MDP)

**Difficulty:** `hard` — Four independent agents acting under LOCAL information only (decentralized, each sees just its own stage + supply line), the canonical hard coordination/bullwhip setting; the classic 36-week instance has NO exact optimum carried (only a reduced 2-agent/3-period self-consistency DP), and the closed-form Sterman 204 is reproduced only by the board-game port while the trainable env.rs yields 378/278 under identical params.

**Verification tier:** `reference` (re-runs a companion / closed-form / reduced-module number, or a published action)

**Tier note:** Split: the closed-form board-game port reproduces the Sterman/Edali-Yasarcan anchor-and-adjust [46,50,54,54]/204 (closed-form, reference-grade) -> headline tier = reference; the trainable env.rs is faithful_unverified (yields 378/278 under identical params, does NOT reproduce 204).

> Status (manifest, verbatim): verified_rerun (closed-form board-game 204 reproduced); faithful_unverified (env.rs as a benchmark, yields 378/278)

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| beer_game_classic_four_stage (PRIMARY) | absent (no field; carries published_sterman_benchmark Some(204) but env.rs does NOT reproduce it: anchor->378, best base-stock->278) | regime:serial_backlog, stages:4, leadtime:ship[2,2,2,2]_order[0,1,1,1], demand:deterministic_step_4to8, horizon:36week, holding:0.5, backlog:1.0, anchor:sterman_S'[28,28,28,20] |
| STERMAN_1989_CLASSIC_BENCHMARK (published row, closed-form only) | absent (notes warn 204 is a property of closed-form bookkeeping only, env.rs yields 378) | published_row:sterman_anchor_adjust, per_agent[46,50,54,54], total:204, scope:closed_form_board_game_only |
| VERIFICATION_PROBLEM_INSTANCE (reduced exact-DP) | false (repo_exact_solver_not_verified_against_literature) | regime:serial_backlog, stages:2, horizon:3periods, discount:0.99, demand:bernoulli{0,1}p=0.5, max_order:4, base_stock:3, scope:repo_internal_only |

## Baselines

**Heuristics**
- base_stock (installation base-stock)
- sterman_anchor_adjust (anchor-and-adjust with targets/adjustment-times/supply-line-weights)

**Exact solver / bound**

finite_horizon_dp.rs bounded backward-induction DP over the reduced 2-agent/3-period VERIFICATION_PROBLEM_INSTANCE; test-only, NOT a Python binding; provably dominates both heuristics on the reduced instance. The classic 4-stage 36-week instance has NO exact optimum carried.

**Published rows**
- Sterman(1989)/Edali-Yasarcan(2014) anchor-and-adjust per-stage [46,50,54,54], total 204 (closed-form board-game only; 204 is the value its R-port emits, NOT a transcribed published table)
- Oroojlooyjadid et al.(2022) Sterman row 45.13 — DELIBERATELY NOT carried (different benchmark, formula/timing do not line up)

## Reference results (compare your approach against these)

_(no learned-policy results recorded for this problem yet.)_

## How to reproduce & compare

**Expected (published) value:** Sterman/Edali-Yasarcan anchor-and-adjust: per-stage [46,50,54,54], total 204 (36-week classic Beer Game)

**Reproduced value (this audit):** closed-form classic_board_game.rs: per-stage [46.0,50.0,54.0,54.0], total 204.0 (EXACT). env.rs under SAME params: sterman_anchor_adjust = 378.0; best base-stock S=24 = 278.0 (env.rs does NOT reproduce 204). Clark-Scarf constant-demand serial-optimum: 4-stage 0.0, 2-stage 0.0 (both reproduced).

**Rerun method / tolerance:** python scripts/decentralized_inventory_control/measure_env_vs_closedform.py -> closed-form [46,50,54,54]/204, env.rs sterman 378, base_stock best S=24->278. Clark-Scarf check via decentralized_inventory_control_policy_rollout_from_paths('base_stock',...): 4-stage->0.0, 2-stage->0.0.

**Reproduce command(s):**

```bash
python /home/nima/code/ml/invman/scripts/decentralized_inventory_control/measure_env_vs_closedform.py
python -c "import invman_rust; print(invman_rust.decentralized_inventory_control_classic_sterman_literature_summary())"
python -c "import invman_rust as r; print(r.decentralized_inventory_control_policy_rollout_from_paths('base_stock',[32.0]*4,[0]*4,[0]*4,[[8,8],[8,8],[8,8],[8,8,8,8]],[[],[8,8],[8,8],[8,8]],[8]*4,[8]*4,[8.0]*4,[8]*4,[8]*100,[0.0]*4,[0.5]*4,[1.0]*4,1.0))"
python -c "import invman_rust; print([n for n in dir(invman_rust) if 'decentralized' in n])"
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._

