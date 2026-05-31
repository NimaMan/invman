# joint-replenishment autoresearch

This is the joint-replenishment counterpart to the dual-sourcing and multi-echelon
autoresearch programs. It targets the `joint_replenishment` problem (Vanvuchelen,
Gijsbrechts & Boute 2020, *Computers in Industry* 119, 103239): two-item replenishment
sharing a full-truckload major order cost, item-specific minor order / holding / shortage
costs, order-before-demand, zero lead time, risk period one. The env is literature-faithful
(Eq. 1-4 verified in-crate; see `rust/src/problems/joint_replenishment/README.md`).

The single-policy loop is the same shape as the sibling programs: train ONE soft-tree
CMA-ES policy on a NAMED instance, evaluate held-out CRN cost + gap vs the strongest
heuristic, append a TSV ledger row. The runner is
`scripts/joint_replenishment/autoresearch_joint_replenishment.py`; it REUSES the
learned-benchmark helpers in `scripts/joint_replenishment/common.py` (the binding
`joint_replenishment_soft_tree_rollout` / `..._population_rollout`).

## Benchmark

The trusted design set is the 16 Vanvuchelen (2020) Table-2 small-scale settings
(`vanvuchelen2020_small_scale_setting_1` .. `_16`). Each is a two-item instance with
truck capacity `V=6`, major cost `K=75`, `d1~U[0,5]`, `d2~U[0,3]`, varying
`(h, b, k)`. They are carried verbatim in `references.rs`.

Strongest heuristic = **MOQ** (`minimum_order_quantity`, evaluated at the per-item
newsvendor order-up-to target `S_i = F_i^{-1}(b_i/(b_i+h_i))`, rounding threshold 2).
DYN-OUT (`dynamic_order_up_to`) is **dominated by MOQ on every one of the 16 settings**,
so MOQ is the sole gap target.

Published anchor = the Figure-3 optimal **action** map for setting 5
(`joint_replenishment_published_action_anchor` /
`VANVUCHELEN_2020_FIGURE3_ANCHOR`): under the optimal policy, in state
`(I1,I2)=(5,0)` only shipper 2 orders, `q=(0,6)` (one full truckload). The paper reports
per-setting optimality only as a figure (Fig. 2: heuristics sit 4-25% above optimal), so
no absolute per-setting optimal cost can be reproduced -- the anchor is an action, not a
cost. The infinite-horizon value-iteration reproduction of that action lives in
`scripts/joint_replenishment/benchmark_vanvuchelen_settings.py`.

Evaluation protocol (inherited from `benchmark_learned_vs_heuristics.py`): held-out
common-random-number seeds from base `1_000_000` (disjoint from the training seed block),
horizon = `periods` (200), discount `gamma = 0.99`. The SAME eval-seed block scores the
learned policy and MOQ (paired / variance-reduced). Mean discounted cost is the metric;
the learned soft-tree and MOQ are on the same scale.

## Intended search surface (the editable levers)

- **Soft-tree structure**: `--depth` (2,3), `--temperature`, `--split_type`
  (oblique / axis_aligned), `--leaf_type` (constant / linear / sigmoid_linear).
- **Action design**: the `vector_quantity` action box per item (currently
  `2 * truck_capacity`) in `common.build_soft_tree_model`; the natural next design is a
  **base-stock-anchored action adapter** so the tree perturbs around the newsvendor
  target instead of emitting raw rounded quantities (this is the high-cost-setting
  recovery lever flagged in the problem README).
- **CMA-ES warm-start at MOQ**: `--warm_start_moq` seeds the CMA mean (`cma_x0`) from a
  vetted candidate rather than a blind random vector, anchoring the search near the
  strongest heuristic's behaviour. (The soft-tree decoder lives in Rust and is not
  analytically invertible into tree params, so the warm-start picks the best of a small
  candidate set -- including the zero vector -- scored on a few training seeds; honest
  decoder-agnostic anchoring, not an exact MOQ encoding.)
- **Deeper budget targeting the high-cost settings**: `--budget full` (depth 3, more
  generations, larger `train_seed_batch`) aimed specifically at settings 2,3,4,8,11,12,15,16.

Files in the surface:

- `scripts/joint_replenishment/` (runner + `common.py` build/eval helpers)
- the soft-tree policy core (`rust/src/core/policies/soft_tree.rs`) and the action
  conversion in `rust/src/problems/joint_replenishment/rollout.rs` (read-only here; the
  Python layer drives structure/action-box via `common.py`)
- the CMA-ES driver (`invman/es_mp.py`)

## Budgets

Defined in `scripts/joint_replenishment/autoresearch_joint_replenishment.py`:

- `smoke`   : popsize 8, 8 generations, train_seed_batch 2, 64 eval seeds (CI / wiring check)
- `screening`: popsize 16, 80 generations, train_seed_batch 4, 512 eval seeds (first pass)
- `full`    : popsize 24, 300 generations, train_seed_batch 12, 2048 eval seeds, depth 3
              default (the high-cost-setting recovery budget)

HARD CPU CAP: the script sets `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` defaults to 2 and
forces `mp_num_processors = 1` (the population rollout path bypasses the multiprocessing
Pool and fans out only via rayon). Two sibling autoresearch agents run in parallel, so the
~27-core default is overridden. Export `RAYON_NUM_THREADS` to change it.

## Goal (keep / discard)

KEEP a candidate (structure + action design + warm-start) if it **beats MOQ** (lower mean
held-out discounted cost, positive `%win`) on the currently-LOSING instances. The headline
target is the high-cost family (`h=5, b=95`): settings 2,3,4,8,11,12,15,16. A candidate
that flips any of those from a loss to a win, without regressing the 6 settings already
won, is promotable. Primary metric: relative gap to MOQ on the same benchmark row
(`gap% = 100*(learned/MOQ - 1)`; negative = learned cheaper = win). Do not lock the search
to one structure -- the job is a policy that beats MOQ on the losers, not to prove soft
trees are universally best.

## What we know (from the learned-benchmark phase, run 2026-05-31)

`benchmark_learned_vs_heuristics.py` (depth 2, oblique, linear leaves, popsize 24, 120
generations, train_seed_batch 4, 2048 held-out CRN seeds) gave: learned **beats MOQ on 6
of 16**, loses on 10. MOQ is the best heuristic on every setting (DYN-OUT dominated).

WINS (learned cheaper than MOQ): setting 5 +13.05%, 13 +11.44%, 14 +6.45%, 6 +4.23%,
9 +1.07%, 1 +0.51% -- the low holding/shortage settings where truckload timing matters.

LOSSES (MOQ cheaper; the targets), `%win` = `100*(MOQ-learned)/MOQ`:

| setting | learned | MOQ | %win |
| --- | ---: | ---: | ---: |
| 2  | 7645.45 | 7186.22 | -6.39% |
| 3  | 8367.58 | 7470.30 | -12.01% |
| 4  | 10197.08 | 8632.45 | **-18.13%** |
| 7  | 9180.72 | 9042.90 | -1.52% |
| 8  | 10833.82 | 10205.05 | -6.16% |
| 10 | 7100.84 | 7058.78 | -0.60% |
| 11 | 8699.72 | 7625.03 | -14.09% |
| 12 | 9838.53 | 8678.75 | -13.36% |
| 15 | 9543.87 | 9268.77 | -2.97% |
| 16 | 11243.73 | 10322.50 | -8.92% |

The biggest losses are the high-cost `h=5, b=95` settings (3,4,8,11,12,16), where ordering
to a tight newsvendor base stock every period is near-optimal and MOQ matches it with less
action variance. A stronger budget (depth 3, 300 generations, train_seed_batch 12) NARROWS
but does not CLOSE these (setting 4 -18.13% -> -13.3%; setting 12 -13.36% -> -10.1%). So the
high-cost gap reflects the **rounded-action soft-tree policy class**, not under-training --
which is why the action-design lever (base-stock-anchored adapter) and the MOQ warm-start
are the priority levers, ahead of pure CMA-ES budget.

Default instance for the runner: `vanvuchelen2020_small_scale_setting_4` (the largest loss,
-18.13%), so a single run lands on the hardest currently-losing target.

## Canonical workspace

Ledger and per-run artifacts:

- `outputs/autoresearch/<run_tag>/results.tsv` (the appended ledger:
  commit, experiment, reference, budget, structure, mean_cost, best_heuristic,
  best_heuristic_name, gap, gap%, winner, description)
- `outputs/autoresearch/<run_tag>/{logs,models}/` (CMA-ES logs + trained soft-tree)
