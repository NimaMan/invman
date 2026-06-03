# Policy Design Guidelines

The objective of this document is to capture, in one reusable place, **how this project
actually designs a policy for an inventory-management problem** so that the resulting
learned-policy claim is honest, comparable to the literature, and reproducible. The recipe
is not "train a soft tree and report a number." It is: build one faithful, correctly-named,
literature-anchored environment first; treat the **action parameterization as part of the
policy** and choose its geometry to match the problem's decision structure; encode the policy
in the best known heuristic's coordinate system so a learned optimizer starts from
something it can at least reproduce; warm-start CMA-ES there and let it refine under paired,
common-random-number evaluation; drive the search through the per-problem `program_<problem>.md`
autoresearch loop; and record everything against a fixed benchmark. These guidelines distill
that recipe from the existing `program_*.md` files (dual_sourcing, multi_echelon,
one_warehouse_multi_retailer, fixed_order_cost, joint_replenishment,
vendor_managed_inventory, lost_sales) and the policy machinery in `invman/policy_build.py`,
`invman/policy_registry.py`, and `invman/dual_sourcing_policy_spec.py`.

---

## 1. Per-problem environment first: anchor the MDP before any learned claim

A learned-policy number means nothing until the environment that produced it is trusted.
Before designing or training any policy:

1. **Build ONE faithful, correctly-named environment per problem.** One env that is
   literature-verified beats several that are approximate. The env home is
   `rust/src/problems/<problem>/` with its own `env.rs::step_state`, `references.rs`,
   `README.md`, and verification. The file name must clearly represent the model
   (e.g. `divergent_special_delivery`, not a generic `multi_echelon`).
2. **Anchor it to published costs BEFORE any learned claim.** Reproduce a published number
   the env must match — a heuristic cost, an exact-DP optimum, or both:
   - exact / finite-horizon DP optimum where it exists
     (`one_warehouse_multi_retailer_exact_dp_summary()` confirms
     `optimal 8.485 <= heuristics 9.2225`; the VMI slice is DP-regression-validated;
     joint_replenishment reproduces the Figure-3 optimal *action*, not a cost, because the
     paper only reports optimality as a figure);
   - published heuristic / constant base-stock costs reproduced to a stated tolerance
     (multi_echelon van_roy_1997 constant base-stock within ~1%; nonstationary_lot_sizing
     reproduces the author's CSV simple and rolling-DP baselines to <=0.17% at 25,000 reps).
   - A **worked-transition unit test** that walks one period by hand against `step_state`
     pins the transition + cost arithmetic independent of any aggregate number.
3. **State honestly when there is no true optimum.** If the literature gives only a strong
   comparator (a DP baseline, a published DRL/PPO row, an action map), say so and make that
   the validation floor — do not dress a strong baseline up as an optimum. VMI has *no*
   published anchor and is labelled `self-consistent-only`; that is reported, not hidden.
4. **Validate the policy forward against the Rust rollout.** Any policy you design is scored
   by the same Rust rollout binding used everywhere else
   (`<problem>_soft_tree_population_rollout`), not by a Python re-implementation that can
   silently drift from `step_state`. The env owns its policy input dimension (ask
   `multi_echelon_policy_feature_dim`, do not re-derive `lw + K*lr` in Python — that formula
   broke on the `lw=0` simple problem).

Only after the env is anchored do learned-policy claims become legitimate, and only then are
they "beats the published number," not "fails to reproduce it."

---

## 2. The central principle: the action parameterization is part of the policy

> **The action parameterization is part of the policy, not an environment restriction.**

Choose the action geometry to match the problem's decision structure rather than accepting a
fixed external grid. A learned policy is only as good as the action space it can express; a
policy that cannot even represent the best heuristic will lose to it. In this repo the action
space is a **policy-DESIGN choice and an autoresearch search dimension**
(`invman/policy_build.py::_build_multi_echelon` says this in code), expressed through the
`action_adapter` / `control_spec` vocabulary in `policy_registry.py` and
`dual_sourcing_policy_spec.py`. The worked precedents:

- **Lost-sales decoders.** The scalar order is decoded through a chosen head:
  `soft_gated_direct_quantity` / `hard_gated_direct_quantity` (a gate logit times a direct
  quantity logit), `soft_gated_ordinal_quantity` (ordinal cumulative head),
  soft-tree leaves (`constant` / `linear` / `sigmoid_linear`). The decoder *is* the policy's
  expressive class; the lost-sales benchmark winner is an oblique depth-2 linear-leaf tree.
- **Fixed-cost gating.** The order/no-order boundary is its own design object: a gate decides
  *whether* to order before the quantity head decides *how much*, so the policy can express
  the `(s,S)` "stay put until you cross `s`" structure that the fixed major cost demands.
  (`fixed_cost_ordinal_stability/README.md` documents when the ordinal head works and fails.)
- **Dual-sourcing factorized capped-dual-index coordinates.** Instead of emitting raw
  `(q_r, q_e)`, the policy emits the best heuristic's parameters `(s_e, Delta_r, cbar_r)` and
  the adapter (`dual_sourcing_capped_dual_index_delta_targets` in
  `dual_sourcing_policy_spec.py::_action_from_controls`) reconstructs the order:
  `expedited = clip(s_e - IP_e, 0, max_e)`, `s_r = s_e + max(Delta_r, 0)`,
  `regular = min(max(0, s_r - IP_r - expedited), cbar_r, max_r)`. The policy now *lives in the
  best heuristic's coordinate system*, so it can reproduce capped-dual-index exactly and then
  refine around it. Control geometry beats raw parameter count.
- **Multi-echelon direct-level estimation.** The Gijs reduced `{50..100}` warehouse grid is an
  "action-space trap": it physically cannot reach the ~300-460 order-up-to level the cost
  structure drives the system to, collapsing the optimum to a degenerate "hold nothing,
  expedite everything" regime (~3090 vs ~911). The fix is **`direct_level`**: estimate the
  warehouse/retailer order-up-to levels directly (continuous -> non-negative int) bounded only
  by the **physical caps** `(Cw, Cr)`, not by a hand-set grid. That single design change moved
  the learned policy from +238% worse to **-14.4% better** than the operating-region best
  constant base-stock, exceeding the published A3C savings.

The recurring lesson across every program file: **action design, not capacity, is the lever.**
When a learned policy loses to a heuristic, suspect the action geometry before reaching for
more optimizer budget.

---

## 3. Decision procedure for choosing the action geometry

For a new problem, choose the action geometry with this procedure:

1. **Identify the known structured heuristic.** Find the strongest in-literature /
   in-repo control: base-stock, echelon base-stock, capped dual index, tailored base-surge,
   MOQ, `(s,S)`, rolling-DP `(s,S)`. This is both the comparator to beat and the coordinate
   system to borrow.
2. **Encode the policy in that heuristic's coordinate system,** so the learned policy can at
   least reproduce it. Build an `action_adapter` whose controls *are* the heuristic's
   parameters (dual-sourcing `(s_e, Delta_r, cbar_r)`; OWMR `symmetric_echelon_targets` =
   one warehouse + one shared retailer target; JR base-stock-anchored adapter perturbing the
   newsvendor target). With a constant leaf at the heuristic's parameter values, **generation
   0 reproduces the heuristic** and the optimizer searches *outward* from a known-good point.
3. **Warm-start CMA-ES at the encoded heuristic** (Section 4), then let it refine. Because the
   leaf transform is non-trivial (`constant: min + sigmoid(p)*span`;
   `linear: min + softplus(bias + w*state)`; see `rust/src/core/policies/soft_tree.rs`), the
   warm-start must **invert the transform** so the leaf actually starts at the heuristic, not
   at a sigmoid-saturated grid maximum. This was a load-bearing OWMR fix: before inverting the
   transform, generation 0 started over-stocked (holdout ~1879 vs heuristic ~1180); after the
   fix, the warm-started constant leaf beat every other variant. Where the decoder is not
   analytically invertible (JR), pick the best of a small candidate set (including the zero
   vector) on a few training seeds — honest decoder-agnostic anchoring, not a fake encoding.
4. **When no good heuristic exists, use direct-level / gated decoders bounded by physical
   caps, not a hand-set grid.** Estimate the order-up-to level or quantity directly, clipped to
   the physical inventory/order caps the env enforces. This is the multi-echelon `direct_level`
   move and the lost-sales direct-quantity move. **Never adopt an external reduced grid as the
   action space without checking it spans the operating region** — confirm the chosen heuristic
   benchmark's argmin is *interior* to its search grid (the multi_echelon setting-2 grid bound
   at `yw=500` and understated the benchmark until the ceiling was raised to ~700).

The single most important lever for beating a near-optimal heuristic is **starting from it**:
on convex / symmetric instances (VMI, OWMR symmetric Poisson(3)) the heuristic is at/near the
optimum, so a random init wastes the budget and a warm-started policy either ties it exactly
or finds the small exploitable deviation.

---

## 4. Optimization: CMA-ES protocol

The optimizer is CMA-ES driving the policy's flat parameter vector, scored by the Rust
population-rollout binding (`invman/cmaes.py` read-only, or `invman/es_mp.py`). Protocol:

- **Two budgets, always.** A `screening` budget (small population, few generations, small
  held-out block) rejects weak ideas fast; a `full` / promotion budget (larger population and
  generations, 2k-4k held-out paths) certifies a winner at decision quality. Budgets are a
  default protocol, not a hard restriction — screen to rank levers, promote to certify.
  Representative full budgets: lost_sales/fixed_cost soft trees; OWMR popsize 32 x 600 gen,
  train_seed_batch 12, 4096 held-out paths; JR popsize 24 x 300 gen, train_seed_batch 12,
  2048 seeds, depth 3; VMI popsize 24 x 200 iters, 4000 soft-tree held-out seeds.
- **Training horizon vs evaluation horizon are distinct.** Train on a (possibly shorter)
  horizon with a warm-up cut for the long-run-average metric (`mean_after_warmup`), then
  **re-evaluate the promoted winner on a long horizon** (`evaluate_saved_policy.py`,
  e.g. `--eval_horizon 1000000`) before trusting the headline number. The fixed-cost winner
  was `8.77528` at `50k` and `8.76576` re-checked at `1M`.
- **Multi-seed / common-random-number paired evaluation, especially when margins are small.**
  Tune the heuristic grid on TRAIN seeds; train the policy on TRAIN seeds; score *all* policies
  on a **disjoint held-out CRN block** with the **same seeds** for learned and heuristic
  (paired / variance-reduced). The argmin heuristic `(W, R)` is re-scored on the disjoint block,
  not reported at its own training argmin. **A sub-stderr "win" is not a win** — watch the
  held-out SEM (OWMR held-out stderr ~1.4-2.4 meant the 0.0% ties were genuine ties, not flips).
  Do not rely on a single training seed; a candidate counts only if it improves the aggregate
  (mean/median/best/worst) across seeds.
- **Warm-starting** as in Section 3: seed the CMA mean (`cma_x0` / `x0`) at the inverted-leaf
  encoding of the strongest heuristic. `sigma_init` controls how far CMA explores around the
  anchor.
- **Divergence / dagger handling.** If the policy diverges or the leaf transform saturates,
  the first suspect is the warm-start encoding (saturated leaf), then the action geometry
  (grid not spanning the operating region), then temperature (too high to express a sharp
  threshold — lower temperature makes the soft tree approach a hard tree). Re-anchor at the
  heuristic and re-screen before adding budget.

---

## 5. Using the autoresearch programs

Each problem family has a `program_<problem>.md` (the agent instructions), a runner under
`scripts/<problem>/autoresearch_<problem>.py`, and a TSV ledger. Use them as follows:

- **One trusted benchmark, one narrow editable surface, one fixed budget, automatic logging,
  keep/discard against a running baseline** (the Karpathy `autoresearch` shape adapted to a
  fixed simulation budget instead of wall-clock). The benchmark instance(s), the evaluation
  harness, and the strongest-heuristic gate are **fixed** — do not edit `references.rs`,
  `reference_instances.py`, or the long-run eval protocol mid-search.
- **Editable search surface:** the soft-tree structure (`--depth {1,2,3}`, `--temperature`,
  `--split_type {oblique, axis_aligned}`, `--leaf_type {constant, linear, sigmoid_linear}`),
  the **action design** (`action_adapter` / `policy_action_mode` / `multi_action_design`),
  the **CMA warm-start** flag, and the budget preset.
- **One policy-focused change per experiment.** Run `--budget screening`, read the ledger,
  keep the change only if it improves the best kept learned cost (or flips a losing instance)
  *and* keeps the code simple; promote survivors to `--budget full`.
- **Keep/discard gate = the in-repo tuned heuristic** (not the published DRL row, which is
  reported but not the gate). Primary metric: held-out relative gap to the best heuristic on
  the same paired CRN block, `gap% = 100*(learned/heuristic - 1)` (or its sign-flipped form);
  **keep** when the sign flips robustly out-of-sample, **discard** designs that stay behind at
  full budget or win only on eval-seed noise.
- **CPU cap.** These loops run several sibling agents in parallel and the bindings otherwise
  grab ~27 cores. Cap at ~4 cores total: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`,
  `mp_num_processors` 1-2, `instance_jobs 1` (the population rollout fans out via rayon, not a
  Python process pool).
- **Record results** in `outputs/autoresearch/<run_tag>/results.tsv` (one row per run:
  commit, experiment, reference, budget, structure flags, mean_cost, best_heuristic, gap, gap%,
  winner, description) plus per-run JSON / logs / models under the same `<run_tag>/`. Promote
  the headline finding into the program file's "What we know" / "Autoresearch outcome" section
  as a prior for the next run, and update `autoresearch/README.md`. Do **not** re-litigate
  established priors.

The autoresearch loop is precisely where "flip a loss to a win via the action box" gets done:
JR setting 10, VMI `low_penalty`, OWMR `partial_backorder`, and the multi_echelon
grid-to-`direct_level` flip are all documented instances of the action-design lever winning
inside an autoresearch run.

---

## 6. Checklist: adding a new problem to the paper

1. **Pick the problem** for a real reason: it must have **literature instances with published
   comparator costs** (an env you can anchor byte-for-byte), and it should add a missing axis
   to the paper.
2. **Build / verify the env** in `rust/src/problems/<problem>/` with `step_state`,
   `references.rs` carrying the published instances **verbatim** (no repo-invented parameters),
   a `README.md`, and a worked-transition unit test. File name represents the model.
3. **Reproduce the published anchor** to a stated tolerance (Section 1). If there is no true
   optimum, state the strong comparator as the validation floor honestly.
4. **Identify the strongest heuristic** and implement it; verify it reproduces its published
   number. This is the keep/discard gate.
5. **Design the action geometry** (Section 2-3): encode the policy in the heuristic's
   coordinate system via an `action_adapter`; if none exists, use direct-level / gated decoders
   bounded by physical caps. Confirm the action space spans the operating region.
6. **Expose the population-rollout binding** (`<problem>_soft_tree_population_rollout`) and wire
   the build helper (`policy_build.py` style) so the env reports its own input dimension.
7. **Warm-start CMA-ES at the heuristic** (invert the leaf transform) and run the
   screening -> full protocol with paired CRN held-out evaluation (Section 4).
8. **Write `program_<problem>.md` and the runner + TSV ledger**, mirroring an existing sibling
   (joint_replenishment is the closest template), and register it in `autoresearch/README.md`.
9. **Report** the headline as a relative gap to the strongest heuristic, paired and held-out,
   with the published DRL/optimum row alongside for context. A flip on a losing instance, a
   beat of the published savings, or an exact tie on a provably near-optimal instance are all
   legitimate results — report what is true.

---

## 7. Worked application of the checklist: `nonstationary_lot_sizing` (next problem)

The freshly-selected next problem is **`nonstationary_lot_sizing`** — the first env in the
paper where demand is **non-i.i.d.** and the policy must react to a rolling forecast. It is
chosen because it is the cleanest "compact ES beats DRL" headline in the candidate set and the
only candidate whose instances and comparator costs are **byte-for-byte published numbers**.
Mapping it onto the checklist:

- **Instances (step 1-2).** The eight Dehaybe, Catanzaro & Chevalier (2024) lost-sales
  rolling-forecast instances carried verbatim in
  `rust/src/problems/nonstationary_lot_sizing/references.rs::LOST_SALES_FORECAST_BENCHMARKS`
  (`dehaybe2024_lostsales_lt2_b5_k10_{constant_5,constant_10,constant_15,seasonal_1,seasonal_2,
  seasonal_4,growth,decline}`): all `L=2, b=5, h=1, K=10`, 104 periods,
  `forecast_horizon=32`, `initial_net_inventory=20`, lost sales. Primary anchor =
  `dehaybe2024_lostsales_lt2_b5_k10_constant_10`. Demand and all params come from the author's
  public testbed (HenriDeh/DRL_MMULS, single-item branch), reproduced verbatim via the 8
  `FORECAST_DEFINITIONS`. Two demand models: CV-Normal (cv=0.2) for the simple/learned/lead-time
  policies, Poisson for the rolling-DP baseline. No repo-invented parameters.
- **Published anchor (step 3).** Two-layer anchoring, already passing: (a) the env+solver
  reproduce both author-CSV baselines (simple and rolling-DP) to `<=0.17%` at 25,000
  replications (`outputs/nonstationary_lot_sizing/learned_benchmark_full8_g150_p48_ac100_r10000.json`:
  simple_pct_diff ~0.05%, dp_pct_diff ~-0.03% on constant_5); (b) the Section-4.2
  worked-transition test (period cost 130) is self-consistent against `env.rs::step_state`.
  There is **no exact optimum** for the rolling-forecast path, so the validation floor is the
  **published DP baseline**, stated honestly — matching the paper's own framing.
- **Strongest heuristic / gate (step 4).** Three structured comparators, all implemented and
  verified: (1) `simple_s_s` (CV-Normal `(s,S)`), reproduced to `<0.05%`; (2) `rolling_dp_s_s`
  (Poisson rolling-DP `(s,S)`), reproduced to `<0.03%` — this is the paper's strong DP baseline
  and the implicit DRL/PPO-target floor, so **beating `rolling_dp_s_s` is the headline**; (3)
  `lead_time_base_stock` (order every period at the lead-time critical ratio), the toughest
  in-repo heuristic on large-demand instances.
- **Action geometry (step 5) — the contribution.** Today the rollout action is **raw 1-D
  direct-order**: `rollout.rs::validate_config` errors unless `action_spec.action_dim == 1`,
  and `action_quantity` returns `action[0]` — the exact analogue of OWMR's weak `direct_orders`
  baseline. The new design is a **forecast-anchored order-up-to residual head**: the leaf emits
  a delta on a forecast-driven target,
  `q = clip(S_hat(forecast_window) - inventory_position + tree_delta, 0, cap)`, where `S_hat`
  is the lead-time-demand critical-ratio level from the normalized forecast window.
  The env already carries `fixed_order_cost` (`env.rs:114`), so an `(s,S)` order/no-order gate
  `q = [s - IP > 0]*(S - IP + delta)` is available as a **second geometry** for the fixed-cost
  regime. Warm-starting the residual head at `delta=0` reproduces the published `(s,S)` /
  base-stock heuristic at generation 0 — the same gen-0-reproduces-heuristic device that worked
  for `symmetric_echelon_targets` (OWMR) and `capped_dual_index` (dual_sourcing).
- **Optimization (step 6-7).** Compact CMA-ES soft-tree, trained directly via the exposed Rust
  binding `nonstationary_lot_sizing_soft_tree_population_rollout` plus read-only `invman.cmaes`
  (the `--learned` path in `scripts/nonstationary_lot_sizing/run_literature_benchmark.py`
  already drives it). Policy state = normalized rolling forecast window + net inventory +
  pipeline. Converged config: depth-2 oblique tree, linear leaves, 150 generations x 48
  candidates, action_cap 100, CRN training seeds disjoint from the 10,000-seed held-out eval
  block. This already produces a converged **8/8 win over the published DP** with the raw
  direct-order action; the new design work is the action head above.
- **Autoresearch + report (step 8-9).** No `program_nonstationary_lot_sizing.md` exists yet
  (the documented gap vs OWMR/JR/VMI). Build a lightweight one mirroring
  `program_joint_replenishment.md`: ledger the head sweep (direct-order vs
  forecast-anchored order-up-to residual vs `(s,S)`-gate residual), CMA warm-start at the
  reproduced `(s,S)` levels, tree depth `{1,2,3}`, leaf `{constant, linear}`, focused on the
  three instances where the converged direct-order policy currently *loses* to
  `lead_time_base_stock` (`seasonal_2` +1.28%, `growth` +0.35%, `decline` +3.46%). Target: flip
  those three to wins via the action head, making it **8/8 single-cheapest** — the same
  "autoresearch flips a loss to a win via the action box" move as JR setting 10 and VMI
  `low_penalty`. The paper story extends the project's "action design, not capacity, is the
  lever" thesis from **static** to **nonstationary, forecast-driven** demand against a
  published DRL-targeted DP baseline.
