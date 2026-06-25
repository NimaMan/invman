# Joint Replenishment — Setting 5 learned policy vs the Value-Iteration optimum

Committed summary of the autoresearch result recorded in
`policy_search/programs/joint_replenishment/README.md` (the full per-path JSON lives at the
gitignored `outputs/autoresearch/joint_replenishment_autoresearch/setting5_vi_optimum_gap.json`).

- **Instance:** Vanvuchelen, Gijsbrechts & Boute (2020) Table-2 **setting 5**
  (`vanvuchelen2020_small_scale_setting_5`): V=6, K=75, k=[40,10], h=[1,1], b=[19,19],
  d1~U[0,5], d2~U[0,3], gamma=0.99.
- **Eval protocol:** 200 periods, init inventory [0,0], **4096 paired common-random-number
  demand paths** (seed base 1_000_000, disjoint from training); the SAME paths score every
  policy (variance-reduced).
- **Baseline = the VALUE-ITERATION OPTIMUM** (the paper's Figure-2 comparator). Independent VI
  over the repo env (converged iter 2260, max delta 9.9e-09) reproduces the published optimal
  action **q=(0,6) at state (5,0)**.
- **Env-arithmetic guard:** the VI policy is rolled out through a Python mirror of
  `env.rs::step_state`; MOQ is rolled out in the same Python env AND via the Rust
  `joint_replenishment_policy_rollout_from_paths`, and they match to **max|diff| = 0.0**, so
  the VI optimality gap is faithful. The learned soft-tree is scored only by the Rust binding
  `joint_replenishment_soft_tree_rollout_from_paths`.

| policy | mean discounted cost (SEM) | vs VI optimum | vs MOQ |
| --- | ---: | ---: | ---: |
| VI optimum (baseline) | 6347.108 (3.34) | — | -16.42% |
| **learned soft-tree (depth-3 oblique, linear leaves, MOQ warm-start)** | **6546.176 (3.64)** | **+3.14%** | **-13.79%** |
| MOQ (strongest heuristic, S=(5,3)) | 7593.655 (4.52) | +19.64% | — |

- **Learned optimality gap = +3.14% above the VI optimum**, closing **84.0%** of MOQ's
  +19.64% gap toward the true optimum (MOQ's gap is inside the paper's Figure-2 4-25% band).
- **Learned beats MOQ by -13.79%, cheaper on all 4096/4096 paired paths** (the autoresearch
  ledger row at 2048 StdRng seeds reads -13.84% — same unanimous win on the independent sampler).
- Learned is cheaper than the VI optimum on only 10/4096 paths (~0.24%, expected ~0), confirming
  VI is the genuine floor.

**Reproduce:**

```bash
# train (≈4.3 min wall, 2 cores)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/autoresearch_joint_replenishment.py \
    --budget full --warm_start_moq --reference vanvuchelen2020_small_scale_setting_5 --seed 123 \
    --description "full: setting5 depth3 oblique linear + MOQ warm-start (VI-optimum-gap paper run)"

# optimality-gap evaluation vs the VI optimum (writes the JSON artifact)
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/joint_replenishment/evaluate_setting5_vs_vi_optimum.py \
    --model_dir outputs/autoresearch/joint_replenishment_autoresearch/joint_replenishment_autoresearch_full_vanvuchelen2020_small_scale_setting_5_d3_oblique_linear_t0.25_wide_moqws_s123/models/joint_replenishment_autoresearch_full_vanvuchelen2020_small_scale_setting_5_d3_oblique_linear_t0.25_wide_moqws_s123_115_300 \
    --eval_paths 4096 \
    --output_json outputs/autoresearch/joint_replenishment_autoresearch/setting5_vi_optimum_gap.json
```
