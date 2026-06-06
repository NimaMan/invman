# Honest-Floor Deploy Rollout — raw-xbest problems (2026-06-06)

## Objective

Apply the OWMR-reference **honest-floor deploy** to every "raw-xbest" problem flagged by the
training-path audit. The floor is an **additive, downside-safe** change at the *deploy* step only:
instead of deploying the CMA-ES single best individual (**xbest** = `es.best_param()` = what
`es_mp.train` returns today), evaluate a candidate set on the **same held-out CRN block** and deploy
the cheaper / higher-return one:

- **xbest** — the historical endpoint (always in the set, so the floor can never be worse).
- **xfavorite** — the CMA-ES distribution mean (`es.current_param()` = `es.result[5]`), the audit's
  diagnosed lever against single-individual overfit to the small training-seed batch.
- **warm-start anchor / gate** — when the runner already carries one (OWMR pattern); never the active
  candidate in practice but kept for strict downside safety.

Guardrails honored on every problem: **no edits to `invman/es_mp.py` or `invman/cmaes.py`** (zero blast
radius / unchanged defaults for all other callers); changes confined to the per-problem runner; a new
`--deploy_endpoint {floor,xbest,xfavorite}` flag (default `floor`) where added, with `xbest`
reproducing the historical deployment **exactly** (verified additive/reversible); no `git commit`
(orchestrator reviews).

## Verdict-change table

| problem | train_path | floor added | re-ran? | old (xbest) → new (floor) seed-mean ± std | verdict change | status |
|---|---|---|---|---|---|---|
| random_yield_inventory | local_train | yes | yes (5) | 228.86 ± 26.34 (+12.32% vs gate, 1/5 below) → **200.99 ± 3.91** (−1.36%, 2/5 below); std −85.2% | **LOSS → PARITY** (downside-safe; no robust beat) | done |
| spare_parts_inventory | es_mp.train | yes | yes (5) | 63.53 ± 6.83 (LOSS, −4.97% vs gate, not all below) → **52.90 ± 0.37** (+12.59%, 5/5 below); std 6.83→0.37 | **LOSS → WIN vs gate** (seed-robust) | done |
| nonstationary_lot_sizing | local_train | yes | yes (8×5) | gap −0.20% ± 5.68% (beat 1/8, **above 2/8**) → gap **−2.41% ± 4.94%** (beat 5/8, **above 0/8**) | **PARITY → robust BEAT** | done |
| joint_pricing_inventory | es_mp.train | yes | yes (5) | profit 220.43 ± 4.91 → **222.87 ± 2.39** (gate 171.59; floor +29.9%, xbest +28.5%); std ~2× tighter | no flip (WIN stays WIN); variance tightened, mean +0.6% | done |
| joint_replenishment | es_mp.train | yes | yes (5) | set5 6569.80 ± 26.29 → 6549.72 ± 34.21 (−13.78% vs gate); set7 9202.68 ± 78.57 → 9162.20 ± 62.09 (+1.32%) | no flip (set5 robust WIN sharpened; set7 robust LOSS tightened) | done |
| vendor_managed_inventory | local_train | yes | yes (5) | 126.42 ± 0.28 (+1.75% vs gate) → 126.30 ± 0.47 (+1.65%); xfavorite deployed 1/5 | no flip (still LOSS); loss narrowed ~0.1pp | done |
| perishable_inventory | local_train | yes | no (reproduces xbest) | smoke: floor == xbest, mean_return −1467.05 (identical, 3/3 seeds) | no change (floor reproduces xbest) | added_no_rerun |
| ameliorating_inventory | local_train | yes | no (reproduces xbest) | smoke: floor == xbest headline 77.668 (xfavorite strictly worse in all 3 smoke runs) | no change (floor reproduces xbest) | added_no_rerun |
| multi_echelon (divergent, Gijs 2022 setting1) | other | yes | yes (5) | best-of-depths 776.15 ± 14.27 (+14.74% vs gate, 5/5 beat) → **identical** 776.15 ± 14.27 | no change (ROBUST_BEAT both ways; design×depth min absorbs per-config xfavorite gain) | added_no_rerun |
| dual_sourcing | other | no | no | not re-run (no optimizer/held-out seam at runner level) | no change (deferred) | deferred |

## Status counts

- **done (floor added + ≥5-seed re-run):** 6 — random_yield_inventory, spare_parts_inventory,
  nonstationary_lot_sizing, joint_pricing_inventory, joint_replenishment, vendor_managed_inventory
- **added_no_rerun (floor added, downside-safe, reproduces xbest / already warm-started):** 3 —
  perishable_inventory, ameliorating_inventory, multi_echelon (divergent)
- **deferred (floor not addable additively at runner level):** 1 — dual_sourcing

**Total: 9 / 10 problems now carry the additive honest floor; 1 deferred.**

## Verdicts changed: 3

Three verdicts flipped, all in the favorable direction and all downside-safe:

1. **random_yield_inventory — LOSS → PARITY.** xbest's single-individual overfit to the small
   training-seed batch produced a heavy held-out tail (e.g. seed 555 = 269.73, +12.32% over gate,
   only 1/5 below). The floor deploys best-of {xbest, xfavorite, LIR-gate}; per-seed deploy was
   {gate, xfavorite, xfavorite, gate, gate}, flooring at the deterministic LIR gate on the seeds where
   even xfavorite overfit. Result: 200.99 ± 3.91, −1.36% vs gate, cross-seed std down 85.2%. Honestly
   **not a robust beat** (2/5 below, gap inside seed noise) — recorded as downside-safe PARITY, not a win.
2. **spare_parts_inventory — LOSS → WIN vs gate (seed-robust).** xfavorite won all 5 seeds; floored
   52.90 ± 0.37 vs gate 60.52 = +12.59%, 5/5 below gate, variance collapsed 6.83 → 0.37.
3. **nonstationary_lot_sizing — PARITY → robust BEAT.** Floor deployed xfavorite on 35/40 runs,
   eliminating all robustly-above-gate instances (constant_10 +7.63% → −1.03%, seasonal_2/4 flipped):
   seed-mean gap −2.41% ± 4.94%, robust beat 5/8, robustly above 0/8.

The other six floored problems showed **no verdict flip** but the floor was uniformly downside-safe:
joint_pricing_inventory and joint_replenishment(set7) **tightened variance**; vendor_managed_inventory
narrowed its loss ~0.1pp; perishable_inventory, ameliorating_inventory, and multi_echelon(divergent)
**reproduced xbest** at the headline (floor never harmful; on multi_echelon it deploys xfavorite
per-config where cheaper, but the runner's design×depth min already lands on the d3 xbest, so the
seed headline is bit-identical to xbest).

## Remaining work (problems still needing the floor or a train-path fix)

- **dual_sourcing — DEFERRED (needs a train-path seam, not a per-problem hack).** Its seed-robust
  runner (`benchmark_full_suite.py` + `aggregate_seed_robust_cdi.py`) funnels every training run
  through the **shared** `invman/experiment_runner.run_experiment`, which calls `train(...)` *without*
  `return_optimizer=True` and scores only xbest. There is no optimizer access and no held-out block
  distinct from the scoring block at the runner level, and no `cma_x0` warm-start vector (CDI anchoring
  lives in the soft-tree leaf init, not as a candidate vector). Adding the floor would require either
  editing shared `run_experiment` (deploys for ~19 other scripts → violates "do not touch other
  problems") or a fragile non-additive rewrite of the suite. **Recommended clean path (separate task):**
  add an additive opt-in `deploy_endpoint` seam *inside* `invman/experiment_runner.run_experiment`
  (default `xbest` = exact current behavior) that reads `es.current_param()` and best-of-evaluates on a
  held-out eval-seed block — one shared change would light up dual_sourcing **and** every other
  `run_experiment` caller (lost_sales, multi_echelon serial, lost_sales_fixed_order_cost, …) at once.
  This is the single remaining train-path fix.
- **multi_echelon (divergent)** already routes through `run_experiment` but got the floor additively
  via a self-contained `_train_with_floor()` in `train_multi_echelon_policy.py` that re-runs
  `run_experiment`'s exact pre-train steps and calls `es_mp.train(return_optimizer=True)` itself — so
  xbest stays bit-identical while xfavorite becomes available without touching the shared module. The
  same pattern (or the shared-seam fix above) is the template for any other `run_experiment`-based
  problem that later needs the floor.

## Zero blast radius

`invman/es_mp.py` and `invman/cmaes.py` were **left untouched** on every problem. All floor logic lives
in the per-problem runners; defaults for all other callers are unchanged. Where a
`--deploy_endpoint {floor,xbest,xfavorite}` flag was added it defaults to `floor`, and
`--deploy_endpoint xbest` reproduces the historical single-endpoint deployment exactly (verified
additive/reversible on every floored problem). No `git commit` was made — orchestrator reviews and
commits.

## Artifacts / cards

- BENCHMARK.md Results sections updated (additive, prior verdicts marked SUPERSEDED where applicable):
  random_yield_inventory, spare_parts_inventory, nonstationary_lot_sizing, joint_pricing_inventory,
  joint_replenishment, vendor_managed_inventory.
- Not updated (no verdict change): perishable_inventory, ameliorating_inventory, multi_echelon.
- Key raw artifacts: `scripts/random_yield_inventory/_floor_5seed.json`,
  `/tmp/nls_5seed_results.json` (nonstationary_lot_sizing).
