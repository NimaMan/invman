# joint_replenishment

Rust-first problem home for `joint_replenishment`.

Repo interpretation:

- multi-item replenishment
- shared full-truckload replenishment cost
- item-specific demand and inventory costs
- feasible raw actions are item order quantities whose aggregate quantity is either zero or an exact
  multiple of the truck capacity

Code lives under `rust/src/problems/joint_replenishment/`.

Verification and benchmark anchors live in:

- `literature/references.rs`
- `verification/tests.rs`
- `practical/`
- `experiments/`

Current status (accurate scope = PARTIAL; audited 2026-05 against the source PDF):

- model fidelity: literature-verified. The env equations match Vanvuchelen et al. (2020) exactly
  (Eq. 1 truck constraint, Eq. 2 cost, Eq. 3 state, Eq. 4 balance, order-before-demand, zero lead
  time, risk period one) -- all confirmed in the paper PDF.
- setting definitions: literature-verified (verbatim). All 16 Table 2 settings match the paper.
- published-number reproduction: PARTIAL / mostly external. The paper reports per-setting optimality
  ONLY as a figure (Figure 2: heuristics 4-25% above optimal), so no absolute optimal-cost number can
  be reproduced. The one exact quotable result is an OPTIMAL ACTION: Vanvuchelen et al. (2020) state in
  prose (Section 6.2, around Figure 3, setting 5) that under the optimal policy, in state
  `(I1,I2)=(5,0)` only shipper 2 orders, `q=(0,6)` (one full truckload), the PPO policy matches it, and
  both heuristics order `q=(2,4)` (quote verified verbatim). This action is carried as
  `VANVUCHELEN_2020_FIGURE3_ANCHOR`. The in-crate tests (`verification/tests.rs`) assert the carried
  anchor's shape AND the env's one-period cost at that stored action (=90 for demand `(2,4)`); they do
  NOT re-derive that `q=(0,6)` is optimal. The optimality reproduction (infinite-horizon value
  iteration, gamma=0.99) is performed only by `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`,
  which is OUTSIDE this crate and not part of `cargo test`. Treat it as faithful-but-external, not an
  in-crate literature assertion.
- repo self-consistency verified: yes on the reduced two-item finite-horizon comparator
  (`finite_horizon_dp.rs`), which confirms the exact DP dominates the carried heuristics on a 4-period
  discounted horizon. This comparator is NOT the paper's infinite-horizon average-cost setting and is
  not asserted against the published action.
- the 16 Vanvuchelen small-scale settings (Table 2) are carried verbatim as public problem
  definitions. The paper reports per-setting optimality gaps only as a figure (Figure 2: the heuristics
  lie 4-25% above optimal), so no full per-setting absolute-cost table is asserted.

State interface:

- `env.rs` exposes raw state quantities only
- `env.rs` validates raw action feasibility, including the full-truckload multiple constraint
- the current soft-tree benchmark keeps any aggregate or normalized policy features in `rollout.rs`
- learned-policy actions are converted to feasible full-truckload quantities in `rollout.rs` before
  entering the environment
- environment code must not hide learned-policy preprocessing

Reference (cited literature):

- Vanvuchelen, Gijsbrechts & Boute (2020), "Use of Proximal Policy Optimization for the Joint
  Replenishment Problem", Computers in Industry 119, 103239.
  DOI: https://doi.org/10.1016/j.compind.2020.103239 (citation verified 2026-05 against Crossref,
  ScienceDirect PII S0166361519308218, and author PDF).
  Open author copy: https://lirias.kuleuven.be/retrieve/badd4d5b-5bfc-44e4-84f1-b98fd113143d
- Model match (faithful): state = previous-period end inventories (Eq. 3); action = order quantities
  with `sum_i q_i = M*V` (Eq. 1); cost `c = sum_i[h_i*I+ + b_i*I- + k_i*1{q_i>0}] + M*K` (Eq. 2);
  order-before-demand (risk period 1); zero lead time; inventory balance `I_t = I_{t-1} + q - d`
  (Eq. 4). All of this is implemented exactly in `env.rs::step_state`.

Benchmark results (reproduced by `scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`):

- Literature anchor (setting 5, infinite-horizon value iteration, gamma=0.99): env-derived optimal
  action at state `(5,0)` is `q=(0,6)`, matching the paper. Confirms env fidelity.
- Repo reduced finite-horizon DP comparator (`VERIFICATION_PROBLEM_INSTANCE`, setting-1 family,
  4 periods, discounted): optimal first action `(6,6)` cost `266.39`; carried MOQ `(7,5)` cost
  `386.10` (gap `+119.71`); carried DYN-OUT `(6,6)` cost `383.96` (gap `+117.57`). Self-consistency
  only.
- Heuristic Monte-Carlo sweep over the 16 Table-2 settings (200 periods, 256 reps, discounted): MOQ
  mean cost ranges ~5990-10303 and DYN-OUT ~6040-10557; MOQ is at or below DYN-OUT on every setting,
  consistent with the paper's finding that `(Q,S|T)` is the stronger heuristic on this small family.

Learned soft-tree vs heuristics (run 2026-05-31 via
`scripts/joint_replenishment/benchmark_learned_vs_heuristics.py`, no Rust rebuild):

- CMA-ES-trained soft-tree (depth 2, oblique splits, linear leaves, popsize 24, 120 generations,
  train_seed_batch 4) evaluated on 2048 held-out common-random-number seeds (eval base 1_000_000,
  disjoint from the training seed block at base 123), 200 periods, gamma 0.99, action box per item
  2*truck_capacity. The same eval-seed block scores the learned policy and both heuristics (paired).
- Result: learned beats the best heuristic (always MOQ here; DYN-OUT is dominated on every setting) on
  6 of 16 settings, loses on 10. Learned WINS on the low holding/shortage settings where truckload
  timing matters: setting 5 +13.0%, setting 13 +11.4%, setting 14 +6.5%, setting 6 +4.2%, plus marginal
  settings 9 (+1.1%) and 1 (+0.5%). Learned LOSES on the high-cost settings (h=5, b=95: settings 3, 4,
  11, 12, 16) by -8.9% to -18.1%, where ordering to a tight newsvendor base stock every period is
  near-optimal and MOQ matches it with less action variance. A stronger budget (depth 3, 300
  generations, train_seed_batch 12) narrows but does not close those losses (setting 4 -18.1% ->
  -13.3%; setting 12 -13.4% -> -10.1%), so the gap on high-cost settings reflects the rounded-action
  soft-tree policy class, not under-training. Full per-setting table and protocol in
  `experiments/README.md`; raw JSON in `outputs/joint_replenishment/learned_vs_heuristics_vanvuchelen.json`.
- On setting 5 (the literature anchor) the learned policy's 13.0% edge over MOQ is consistent with the
  paper's Figure-2 finding that the heuristics sit 4-25% above optimal.

Remaining steps:

- Newly added `VANVUCHELEN_2020_FIGURE3_ANCHOR` and the `joint_replenishment_published_action_anchor`
  binding are callable from the installed `invman_rust` (used by both benchmark scripts).
- A stronger learned policy class for the high-cost settings (e.g. a base-stock-anchored action adapter
  so the soft-tree perturbs around the newsvendor target instead of emitting raw rounded quantities)
  is the natural next experiment to recover the -8% to -18% high-cost-setting losses.
