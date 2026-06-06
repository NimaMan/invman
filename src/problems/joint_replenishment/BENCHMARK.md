# joint_replenishment — benchmark card

**One-line MDP:** state = previous-period end inventories of 2 items; action = per-item order
quantities whose total is `0` or an exact multiple of truck capacity `V`; one-period cost =
`M*K + sum_i[k_i*1{q_i>0} + h_i*(I_i)^+ + b_i*(I_i)^-]`; objective = minimize infinite-horizon
discounted cost (gamma = 0.99).

**Status:** verified_rerun of a published *ACTION* (q=(0,6) re-derived by value iteration); no
published absolute cost table exists, so all cost numbers are repo-native.
**Paper:** §related-work / §future-work only of learning_inventory_control_policies_es.tex
(`vanvuchelen2020jrp`, lines 248 and 3808 + bibliography 3970). There is NO dedicated experiment
section for joint_replenishment in the paper; it is cited as context for structured-policy and
generalist directions.

## Problem formulation
Source model: Vanvuchelen, Gijsbrechts & Boute (2020), "Use of Proximal Policy Optimization for the
Joint Replenishment Problem", Computers in Industry 119, 103239 (DOI 10.1016/j.compind.2020.103239;
open author copy https://lirias.kuleuven.be/retrieve/badd4d5b-5bfc-44e4-84f1-b98fd113143d).

- **Timing (risk period one, zero lead time, order-before-demand):** at the start of period `t` the
  state is the previous-period ending inventory vector. The agent places orders `q_t`, they arrive
  immediately, then demand `d_t` is realized.
- **State:** `I_{t-1}` = inventory levels per item (2 items); env raw state also carries the period
  index (`env.rs::build_raw_state`).
- **Action:** order quantities `q = (q_1, q_2)` with the full-truckload constraint `sum_i q_i = M*V`
  for integer `M >= 0` (zero, or an exact multiple of `V`); `M` = trucks dispatched
  (`env.rs::trucks_required` rejects any non-multiple).
- **Transition (Eq. 4 balance):** `I_t,i = I_{t-1},i + q_i - d_i`.
- **One-period cost (Eq. 2):** `period_cost = M*K + sum_i[ k_i*1{q_i>0} + h_i*max(I_t,i,0)
  + b_i*max(-I_t,i,0) ]`, where `K` = major (per-truck) cost, `k_i` = minor (per-item) order cost,
  `h_i` = holding, `b_i` = backorder. Implemented exactly in `env.rs::step_state`; reward = `-period_cost`.
- **Objective:** long-run infinite-horizon discounted cost, gamma = 0.99 (the paper's setting). The
  in-crate `finite_horizon_dp.rs` is a separate 4-period discounted self-consistency comparator, NOT
  the infinite-horizon objective.

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| vanvuchelen2020_small_scale_setting_5 (PRIMARY + Figure-3 anchor) | regime:backorder; items:2; truck V=6; K=75; minor k=[40,10]; holding h=[1,1]; shortage b=[19,19]; demand U[0,5]xU[0,3]; gamma 0.99; has VI optimum | V=6, K=75, k=[40,10], h=[1,1], b=[19,19], d1~U[0,5], d2~U[0,3] | absent (no `literature_verified` field; carried verbatim from Table 2; the optimal ACTION is verified) |
| vanvuchelen2020_small_scale_setting_1..16 (16 Table-2 settings) | regime:backorder; items:2; V=6; K=75; h in {1,5}; b in {19,95}; minor k in {[10,10],[40,10]}; gamma 0.99 | per-setting (h,b,k) sweep over Table 2; demand U[0,5]xU[0,3] | absent (every h/b/k/K/V/demand value matches Table 2; NO per-setting absolute cost reproduced) |
| VERIFICATION_PROBLEM_INSTANCE (reduced 4-period DP) | regime:backorder; items:2; periods:4; self-consistency only | setting-1 family k=[10,10], h=[1,1], b=[19,19], V=6, K=75, init [2,0], gamma 0.99 | false (`repo_finite_horizon_self_consistency_comparator`) |

## Baselines
- **Heuristics:** `minimum_order_quantity` (MOQ / (Q,S|T), evaluated at per-item newsvendor
  order-up-to target `S_i = F_i^{-1}(b_i/(b_i+h_i))`, rounding threshold 2) — strongest on all 16
  settings; `dynamic_order_up_to` (DYN-OUT) — dominated by MOQ on every one of the 16 settings.
- **Exact / optimal:** TWO solvers. (1) in-crate reduced finite-horizon DP (`finite_horizon_dp.rs`,
  4-period, self-consistency only); (2) infinite-horizon discounted value iteration specialised to
  setting 5 (in `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`), which re-derives
  the published Figure-3 optimal ACTION q=(0,6) at state (5,0) and yields a setting-5 VI-optimum mean
  cost of 6347.108 (repo-native, NOT a published number). The paper gives NO absolute optimal-cost table.
- **Published comparators (CONTEXT only):** Figure-3 optimal ACTION (setting 5, state (5,0)) =
  q=(0,6) — the one exact published quantity, re-derived by VI. Figure-3 heuristic ACTION = q=(2,4) is
  a STORED LITERAL only; the repo MOQ actually orders (0,6), NOT (2,4), so this stored literal is a
  caveat, not a reproduction. Figure 2 reports heuristics 4-25% above optimal as a figure with no
  extractable per-setting numbers. The paper's PPO costs are NOT carried (cross-protocol DRL).

## Verification
- Published number: **an ACTION** — setting-5 Figure-3 optimal q=(0,6) at state (5,0); the paper
  publishes NO absolute cost.
- **Re-run reproduced: VI greedy action at (5,0) = (0,6)** (converged iter 2260, max delta 9.92e-09)
  via
  `python -c "import sys; sys.path.insert(0,'scripts/joint_replenishment'); import benchmark_vanvuchelen_settings as bvs; g,it,d=bvs.value_iteration_setting5(lo=-12,hi=18); print(tuple(int(x) for x in g((5,0))[0]), it, d)"`
  and the finite-horizon DP self-consistency comparator
  `python -c "import invman_rust as ir; print(ir.joint_replenishment_exact_dp_summary())"` which
  matches the README literals: optimal (6,6) = 266.386, MOQ (7,5) = 386.101, DYN-OUT (6,6) = 383.960.
  Verdict: **verified_rerun of a published action**; cost numbers are repo-native.
- **Debts/caveats:** (1) the published quantity is an ACTION, not a cost — there is no peer-reviewed
  cost table to reproduce, so this is NOT a verified-against-a-paper-cost system. (2) The setting-5
  VI-optimum mean cost 6347.108 is faithful but its JSON/model artifact is gitignored and was NOT
  re-run in this audit (the q=(0,6) action derivation WAS re-run). (3) The Figure-3 heuristic literal
  q=(2,4) is a stored snapshot the repo MOQ contradicts (repo MOQ orders (0,6)); treat (2,4) as a
  verbatim paper quote, not a repo behavior. (4) The 4-period DP is self-consistency only and is NOT
  the paper's infinite-horizon average/discounted objective.

## Results (learned policy)
All learned results below are at_risk = single optimizer seed or best-of-N and are **NOT yet
seed-robust** by the repo standard (mean ± std over >= 5 optimizer seeds vs the same-protocol gate):

- **CMA-ES soft-tree beats MOQ on 6 of 16 settings (single optimizer seed 123, NOT seed-robust):**
  setting 5 +13.05%, 13 +11.44%, 14 +6.45%, 6 +4.23%, 9 +1.07%, 1 +0.51% (cheaper = positive).
  Loses on 10. Held-out 2048 paired CRN seeds (eval base 1_000_000, disjoint from training), 200
  periods, gamma 0.99.
- **Setting 5 vs the VI optimum (single seed 123, NOT seed-robust):** learned soft-tree (depth-3
  oblique, linear leaves, MOQ warm-start) 6546.176 (SEM 3.64) = +3.14% above the VI optimum 6347.108
  (SEM 3.34), closing 84.0% of MOQ's +19.64% gap; learned beats MOQ -13.79%, cheaper on all
  4096/4096 paired paths. (4096-path eval; env-arithmetic guard max|diff| 0.0 between Python and Rust
  MOQ rollout.)
- **Setting 10: flipped loss to WIN, gap -0.85% (best_of_n / 2 seeds, NOT seed-robust):** `d3 oblique
  linear basestock slack1 + MOQ warm-start` learned 6998.6 vs MOQ 7058.8; "robust across two seeds"
  s123 -0.79% / s777 -0.85% — this is 2 seeds, below the >= 5-seed bar.
- **Setting 7: closed to +0.09% near-tie (best of 7 seeds, NOT seed-robust):** never strictly flips;
  remains a loss to MOQ (gap ranged +0.09% to +1.85% across 8 seeds).
- **Honest downside-safe floor + seed-robust (5 optimizer seeds, training-path audit 2026-06-06):**
  `autoresearch_joint_replenishment.py` now supports `--deploy_endpoint {floor,xbest,xfavorite}`
  (default `floor`, ADDITIVE; `xbest` reproduces the historical single endpoint exactly). The floor
  deploys the best-of {xbest = `es.best_param()`, xfavorite = CMA distribution mean `es.current_param()`
  = `result[5]`, warm-start anchor `cma_x0` when `--warm_start_moq`} on the SAME held-out block — it
  never deploys worse than xbest. Setting 5 (`d3 oblique linear` + MOQ warm-start), seeds 9001–9005,
  full budget (pop24/300gen/batch12), 2048 held-out CRN seeds: **xbest 6569.80 ± 26.29 (−13.52% vs MOQ
  gate 7596.67); floored 6549.72 ± 34.21 (−13.78%)** — floor deviated to xfavorite on 4/5 seeds,
  reproduced xbest on 1/5 (downside-safe), 5/5 below the gate for both endpoints. Verdict unchanged
  (robust WIN, sharpened). Setting 7 (`d3 basestock` + warm-start, same 5 seeds): xbest 9202.68 ± 78.57
  (+1.77%) → floored 9162.20 ± 62.09 (+1.32%, std tightened); 0/5 below gate — remains a robust LOSS
  (floor helps but does not flip). Reproduce: append `--deploy_endpoint floor` (default) to recipe #5.
- **Structural loss (single seed, at_risk = false):** MOQ dominates DYN-OUT on all 16; the learned
  soft-tree LOSES on the high-cost h=5,b=95 family (settings 3,4,11,12,16) by -8.9% to -18.1% even at
  depth-3 / 300-generation budget — a rounded-action policy-class limit, not under-training. The
  flagged native fix is a Rust base-stock-residual action head `order = clip(S_i - I_i + tree_delta)`.

## Reproduce
```bash
# 1. published-action verification: finite-horizon DP self-consistency summary
python -c "import invman_rust as ir; print(ir.joint_replenishment_exact_dp_summary())"

# 2. re-derive the published Figure-3 optimal action q=(0,6) at state (5,0) by value iteration
cd /home/nima/code/ml/invman && python -c "import sys; sys.path.insert(0,'scripts/joint_replenishment'); import benchmark_vanvuchelen_settings as bvs; g,it,d=bvs.value_iteration_setting5(lo=-12,hi=18); print(tuple(int(x) for x in g((5,0))[0]), it, d)"

# 3. inspect the carried published-action anchor
python -c "import invman_rust as ir; print(ir.joint_replenishment_published_action_anchor())"

# 4. learned soft-tree vs heuristics over the 16 Table-2 settings (2-core cap)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/benchmark_learned_vs_heuristics.py

# 5. autoresearch single-policy search (full budget, MOQ warm-start, setting 5)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/autoresearch_joint_replenishment.py --budget full --warm_start_moq --reference vanvuchelen2020_small_scale_setting_5 --seed 123

# 6. setting-5 learned vs VI optimum (4096 paths)
RAYON_NUM_THREADS=2 python scripts/joint_replenishment/evaluate_setting5_vs_vi_optimum.py --eval_paths 4096
```

## Pointers & caveats
- code: `src/problems/joint_replenishment/env.rs` (MDP), `finite_horizon_dp.rs` (4-period DP),
  `literature/references.rs` (`VANVUCHELEN_2020_FIGURE3_ANCHOR`, 16 `SMALL_SCALE_SETTINGS`,
  `VERIFICATION_PROBLEM_INSTANCE`), `bindings.rs`
  (`joint_replenishment_exact_dp_summary`, `joint_replenishment_published_action_anchor`),
  `verification/` (in-crate tests), `rollout.rs` (learned-action feasibility decode).
- scripts: `scripts/joint_replenishment/` (`benchmark_vanvuchelen_settings.py` = VI;
  `benchmark_learned_vs_heuristics.py`; `autoresearch_joint_replenishment.py`;
  `evaluate_setting5_vs_vi_optimum.py`; `setting5_vi_optimum_gap_result.md`).
- autoresearch: `autoresearch/program_joint_replenishment.md`.
- Honest caveats: (a) the only published quantity is an ACTION q=(0,6), not a cost — this is
  verified_rerun of an action, NOT against a paper cost table. (b) The paper's PPO costs are
  cross-protocol DRL and are not carried/compared. (c) Figure-3 heuristic literal q=(2,4) is a paper
  quote that repo MOQ does not reproduce (repo MOQ orders (0,6)). (d) ALL learned win claims above are
  single-seed or best-of-N and are NOT yet seed-robust (need mean ± std over >= 5 optimizer seeds vs
  the same-protocol MOQ gate). (e) Demand is uniform U[0,high] (range, not std). (f) An existing
  `README.md` in this folder predates this card and is consistent with it; this BENCHMARK.md does not
  replace it.
