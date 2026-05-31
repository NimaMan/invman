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

Autoresearch (single-policy policy-search loop, mirrors dual_sourcing / multi_echelon):

- The learned soft-tree LOSES to MOQ on 10 of the 16 settings (the high-cost `h=5, b=95`
  family: settings 2,3,4,8,11,12,15,16, plus marginal 7,10). The autoresearch loop searches the
  soft-tree policy design for one that BEATS MOQ on those losers. The program file is
  `autoresearch/program_joint_replenishment.md` (trusted benchmark = the 16 Table-2 settings;
  strongest heuristic = MOQ, DYN-OUT dominated; published anchor = the Fig-3 optimal action
  `q=(0,6)` at state `(5,0)`; editable levers = tree depth/temperature/split/leaf, the action box /
  base-stock-anchored action adapter, CMA-ES warm-start at MOQ, and a deeper depth-3 budget targeting
  the high-cost settings; keep-rule = beat MOQ on a currently-losing setting without regressing the
  6 wins).
- The runner is `scripts/joint_replenishment/autoresearch_joint_replenishment.py`. It REUSES the
  learned-benchmark helpers in `scripts/joint_replenishment/common.py` (binding
  `joint_replenishment_soft_tree_rollout` / `..._population_rollout`): it trains ONE soft-tree with
  CLI-selected structure on a NAMED instance (default `vanvuchelen2020_small_scale_setting_4`, the
  -18.13% worst loser), evaluates held-out common-random-number cost + gap vs MOQ (paired eval block
  from base 1_000_000), and APPENDS a TSV ledger row (cost, MOQ, gap, gap%, winner) under
  `outputs/autoresearch/<run_tag>/results.tsv`. Budgets: `smoke` / `screening` / `full` (full =
  depth-3, the high-cost-setting recovery budget). Run under a hard 2-core cap (sibling agents run in
  parallel; the script sets `RAYON_NUM_THREADS`/`OMP_NUM_THREADS` defaults to 2 and forces
  `mp_num_processors=1`):

  ```
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/autoresearch_joint_replenishment.py \
      --budget screening --warm_start_moq --depth 3 \
      --reference vanvuchelen2020_small_scale_setting_12 \
      --description "screening: depth3 + MOQ warm-start on high-cost loser"
  ```

Autoresearch outcome (focused full-budget policy search, run 2026-05-31; no Rust rebuild):

- Lever implemented at the Python action-box layer (the Rust `vector_quantity` decoder is read-only):
  a **base-stock-anchored action box** (`--action_box basestock --cap_slack S`) that caps each item's
  order box at `newsvendor_target_i + S` instead of the wide `2*truck_capacity` box. This is the
  flagged "base-stock-anchored action adapter": for the high-cost family the newsvendor target equals
  `demand_high_i`, so the same tree-output range maps onto a tight band around the optimal base-stock
  order, finening decode resolution. Implemented in `common._max_order_quantities` /
  `build_soft_tree_model` and wired through the runner CLI (`--action_box`, `--cap_slack`); the box tag
  (`wide` / `bsN`) is recorded in the ledger `policy_architecture` column. All runs warm-started at MOQ.
- Search coverage: ~25 configurations on the three closest-to-flipping losers (settings 7, 10, 15),
  full budget (popsize 24, 300 generations, train_seed_batch 12, depth 3 default), 2048 held-out CRN
  eval seeds, 2-core cap. Levers swept: action box (wide / basestock slack 0,1,2), depth (2,3),
  split (oblique / axis_aligned), temperature (0.1, 0.25), sigma_init (1.5, 2.5), and a seed ensemble
  (123, 202, 314, 42, 777, 909, 1234) on the best class. The larger high-cost losers (4, 11, 12) were
  NOT re-searched: the learned-benchmark phase already showed those reflect the rounded-action policy
  class limit (-13% to -18% even at full budget), so the promotable targets are the marginal losers.
  Best config + every run are in the ledger `outputs/autoresearch/joint_replenishment_autoresearch/results.tsv`.
- Results vs MOQ (held-out, 2048 CRN seeds; gap% = 100*(learned/MOQ - 1), negative = learned cheaper):
  - **setting 10: FLIPPED to a WIN.** `d3 oblique linear basestock slack1` -> learned 6998.6 vs MOQ
    7058.8, gap **-0.85%** (robust: seed 123 -0.79%, seed 777 -0.85%). The base-stock-anchored box is
    the decisive lever here (wide box stays at +2.9%; slack0 too tight at +4.1%; slack2 regresses to
    +2.2%, so slack1 is the sweet spot). Benchmark was -0.60% (MOQ ahead); now learned ahead.
  - setting 7: closed from -1.52% to **+0.09%** (essentially tied at MOQ), best = `d3 oblique linear
    wide` seed 202 -> learned 9050.7 vs MOQ 9042.9. Across 8 seeds the gap ranged +0.09% to +1.85%; the
    best basin sits within evaluation noise of MOQ but never strictly flips. The basestock box did not
    beat the wide box here (best basestock +0.61%).
  - setting 15: closed from -2.97% to **+2.45%** (best = `d3 oblique linear wide`), not flipped. This
    setting has a high minor-order cost (40) on item 1 with asymmetric `h=(5,1) b=(95,19)`, which
    favours MOQ's low-action-variance batching; the basestock box hurts here (+4.1% to +10.9%).
- Keep / discard gate (beat MOQ on a loser without regressing the 6 wins): setting 10's
  `d3 oblique linear basestock slack1 + MOQ warm-start` PASSES (a clean flip). Settings 7 and 15 remain
  losses (7 a near-tie, 15 a structural MOQ-favouring instance). The base-stock-anchored action box is
  a promotable, opt-in lever (it flips the marginal high-cost-near-symmetric loser, but regresses the
  high-minor-cost asymmetric loser, so it is not made the default).

Remaining steps:

- Newly added `VANVUCHELEN_2020_FIGURE3_ANCHOR` and the `joint_replenishment_published_action_anchor`
  binding are callable from the installed `invman_rust` (used by both benchmark scripts).
- The deepest high-cost losses (settings 4, 11, 12, 16: -13% to -18%) still reflect the rounded-action
  soft-tree policy class. A native (Rust-side) base-stock-residual action head -- order = clip(S_i - I_i
  + tree_delta) rather than a clipped raw box -- is the next lever to recover those; the Python action
  box only narrows the decode range, it cannot remove the integer-rounding floor on the residual.
