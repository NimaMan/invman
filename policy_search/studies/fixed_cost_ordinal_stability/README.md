# Fixed-Cost Ordinal Stability

This note tracks one narrow question:

- when does the fixed-cost ordinal head work,
- when does it fail,
- and why did the recent `L=4, p=4, K=5` Rust rerun look much worse than the older trusted
  numbers?

The policy family of interest is the ordinal quantity head:

- old alias: `linear_gated_ordinal_quantity`
- current canonical name: `linear_soft_gated_ordinal_quantity`

These names are the same family in the repo:

- Python alias normalization:
  - `invman/policy_registry.py`
- Rust parser alias normalization:
  - `src/core/policies/dense.rs`

## Regression table

The controlled retrains are now complete. The table below is the clean answer to "where do we
lose performance?"

All three rows use the same canonical fixed-cost benchmark family:

- problem: `lit_pois_mu5_l4_p4_k5`
- backend: Rust
- policy family: ordinal linear head
- `Q=50`
- `training_episodes=5000`
- evaluation: `10` seeds, horizon `10^6`

Only the state semantics differ.

| Run | State semantics | Mean cost | Std. dev. | Gap vs archived winner |
| --- | --- | ---: | ---: | ---: |
| Archived canonical winner | old `pipeline` meaning, effectively `state / Q` | 8.7750 | 0.0070 | 0.0000 |
| Exact retrain on current code | `pipeline = state / demand_mean` | 10.6305 | 0.0061 | +1.8555 |
| Exact retrain with restored old scaling | `pipeline_qscaled = state / Q` | 8.7734 | 0.0076 | -0.0016 |

Files:

- archived winner:
  `outputs/benchmarks/fixed_cost_canonical_suite_5k_seed42/results/fixed_cost_canonical_suite_5k_seed42_linear_gated_ordinal_quantity.json`
- bad retrain:
  `outputs/benchmarks/fixed_cost_exact_ordinal_rust_current_pipeline_seed42/results/fixed_cost_exact_ordinal_rust_current_pipeline_seed42_linear_soft_gated_ordinal_quantity.json`
- recovered retrain:
  `outputs/benchmarks/fixed_cost_exact_ordinal_rust_qscaled_pipeline_seed42/results/fixed_cost_exact_ordinal_rust_qscaled_pipeline_seed42_linear_soft_gated_ordinal_quantity.json`

This table is the main regression summary:

- current `pipeline` loses about `1.86` cost units
- restoring the old `Q`-scaled state fully recovers the old basin
- so the regression is in the state semantics, not in CMA-ES itself and not in the policy family

## Checkpoint ablation table

The archived good checkpoint was also evaluated under several environment variants to isolate
which semantic change actually kills it.

| Archived good checkpoint evaluated under | Mean cost |
| --- | ---: |
| current semantics | 14.8167 |
| old scale only | 8.7545 |
| old init only | 14.8569 |
| old scale + old init | 8.7455 |
| full old env | 8.7501 |

This isolates the dominant factor cleanly:

- changing only the initialization does almost nothing
- restoring the old scale almost fully restores performance
- therefore the state scaling drift is the killing factor

## Earlier intermediate recovery run

Before the controlled `pipeline` versus `pipeline_qscaled` retrains above, there was an earlier
recovery attempt on the same fixed-cost family:

- run root:
  `outputs/benchmarks/fixed_cost_l4_p4_k5_q50_pop64_soft_gated_ordinal_rust_seed42`
- log:
  `outputs/benchmarks/fixed_cost_l4_p4_k5_q50_pop64_soft_gated_ordinal_rust_seed42/logs/log_fixed_cost_l4_p4_k5_q50_pop64_soft_gated_ordinal_rust_seed42_linear_soft_gated_ordinal_quantity.txt`
- status:
  `outputs/benchmarks/fixed_cost_l4_p4_k5_q50_pop64_soft_gated_ordinal_rust_seed42/results/status_fixed_cost_l4_p4_k5_q50_pop64_soft_gated_ordinal_rust_seed42_linear_soft_gated_ordinal_quantity.json`

Snapshot taken on 2026-04-04:

- around `e876`
- best reward about `-10.06`
- population mean about `-10.68`

That run is now superseded by the controlled retrains in the regression table above. It is kept
here only as historical context.

## What the archive says

The result archive is not consistent with the claim that this policy is simply random.
It is much more structured than that.

### Where the ordinal head works

Trusted good Rust and Python results exist for low fixed-cost, low shortage-cost Poisson cases
with `Q=50`.

Strong examples:

- Rust, `L=1, p=4, K=5`, `training_episodes=2000`, `pop=50`:
  - `8.20781`
  - file:
    `outputs/benchmarks/fixed_cost_full_grid_suite_2k_pop50_h2000_seed42_status/results/fixed_cost_full_grid_suite_2k_pop50_h2000_seed42_status_lit_pois_mu5_l1_p4_k5_linear_soft_gated_ordinal_quantity.json`
- Rust, `L=1, p=4, K=5`, `training_episodes=2000`, `pop=64`:
  - `8.22194`
  - file:
    `outputs/benchmarks/fixed_cost_full_grid_suite_2k_pop64_h2000_seed42/results/fixed_cost_full_grid_suite_2k_pop64_h2000_seed42_lit_pois_mu5_l1_p4_k5_linear_soft_gated_ordinal_quantity.json`
- Rust, `L=2, p=4, K=5`, `training_episodes=2000`, `pop=64`:
  - `8.56017`
  - file:
    `outputs/benchmarks/fixed_cost_full_grid_suite_2k_pop64_h2000_seed42/results/fixed_cost_full_grid_suite_2k_pop64_h2000_seed42_lit_pois_mu5_l2_p4_k5_linear_soft_gated_ordinal_quantity.json`
- Rust canonical promoted run, `L=4, p=4, K=5`, `training_episodes=5000`:
  - `8.77502`
  - file:
    `outputs/benchmarks/fixed_cost_canonical_suite_5k_seed42/results/fixed_cost_canonical_suite_5k_seed42_linear_gated_ordinal_quantity.json`
- Python full-grid paperlike run, `L=4, p=4, K=5`, `training_episodes=5000`, `pop=50`:
  - `8.77228`
  - file:
    `outputs/benchmarks/fixed_cost_full_grid_suite_5k_paperlike/results/fixed_cost_full_grid_suite_5k_paperlike_lit_pois_mu5_l4_p4_k5_linear_gated_ordinal_quantity.json`

### Where the ordinal head fails

The same family degrades sharply when the shortage cost or fixed ordering cost is higher.

Examples:

- Rust, `K=5`, `p=19`:
  - `L=1`: about `11.06`
  - `L=2`: about `12.40`
- Rust, `K=25`, `p=4`:
  - around `20.00`
- Rust, `K=25`, `p=19`:
  - about `18.59` to `19.58`

So the policy is not uniformly robust. It is strong on the easier low-`K`, low-`p` Poisson
regime and weak on the harder fixed-cost regimes.

## Main findings

### 1. The policy is regime-sensitive, not random

The archive pattern is stable:

- good on `K=5, p=4`
- worse on `K=5, p=19`
- much worse on `K=25`

That already explains part of the “sometimes works, sometimes not” behavior.

### 2. The recent bad `L=4, p=4, K=5` reruns are also affected by state drift

The most important regression is not `pop=64` versus `pop=50`. It is that the canonical
`pipeline` state changed in commit `8b420dd` (`Decouple lost-sales env actions from order caps`).

That commit changed:

- the state scale from `state / max_order_size` to `state / demand_mean`,
- the initial lead-time orders from random `1..Q` to deterministic `round(demand_mean)`,
- the env action validity from `0..Q` to unbounded nonnegative.

For the canonical fixed-cost instance:

- old scale: divide by `Q=50`
- current scale: divide by `demand_mean=5`

So the same physical pipeline vector is now presented to the model at `10x` larger magnitude.

This is enough to destroy the old ordinal policy. Using the archived good checkpoint from
`fixed_cost_canonical_suite_5k_seed42`, the following short-horizon ablation on current code gives:

- current state semantics: `14.8167`
- old scale only: `8.7545`
- old init only: `14.8569`
- old scale + old init: `8.7455`
- full old env: `8.7501`

So the dominant killing factor is the state scaling change. The initialization change is nearly
irrelevant for this benchmark once the old scale is restored.

The action-cap change is also not the main issue for this particular policy family, because the
ordinal decoder already clips to `max_order_size` at the policy level.

Checkpoint-level evidence points in the same direction. On the same physical state, the archived
good model behaves very differently under old and new scaling:

- state `[5, 0, 0, 0]`: old scaling orders `12`, current scaling orders `0`
- state `[25, 0, 0, 0]`: old scaling orders `13`, current scaling orders `0`

### 3. The recent bad `L=4, p=4, K=5` Rust rerun is not comparable to the best old result

The recent recovery run was:

- Rust
- `Q=50`
- `pop=64`
- `training_episodes=2000`

The old trusted `L=4` good runs used a promoted budget:

- Rust canonical: `training_episodes=5000`
- Python full-grid paperlike: `training_episodes=5000`, `pop=50`

So even after accounting for state drift, the recent miss is not “same protocol, different
outcome.” The training budget is different too.

This matters because the easier `L=1` and `L=2` cases already succeed at `2000`, while the `L=4`
case appears to need the promoted budget.

### 4. Population size is not the main explanation

Both `pop=50` and `pop=64` succeed on the easy `K=5, p=4` fixed-cost cases for `L=1` and `L=2`.
So the ordinal head is not simply “good at 50 and bad at 64.”

The stronger explanation is:

- easy instances work at the screening budget,
- harder `L=4` needs the promoted budget,
- harder `p` or `K` regimes fail even with the same family.

### 5. Python and Rust are not semantically identical for the ordinal head

The ordinal decoder has backend drift.

Python:

- current Python policy descriptor/registry path: `invman/policy.py`, `invman/policy_registry.py`
- `soft_gated_ordinal_quantity` does:
  - `round(sigmoid(g) * sum(sigmoid(o_k)))`
  - then clips to `[0, max_order_size]`

Rust:

- `src/core/policies/dense.rs`
- `SoftGatedOrdinalQuantity` does:
  - `round(sigmoid(g) * sum(sigmoid(o_k)))`
  - without an explicit final clamp

For `Q=50` this is mostly harmless because the score is already bounded by the number of ordinal
logits, but it still means the backends are not strictly identical and should not be treated as
interchangeable without checking.

### 6. Old “good ordinal” references mixed two kinds of baselines

We have been using two different kinds of “good old result”:

- Rust canonical promoted run
- Python full-grid paperlike run

Both are useful, but they answer different questions. The current failure only becomes interpretable
once the protocol and backend are matched.

## Working interpretation

The current best interpretation is:

1. `linear_soft_gated_ordinal_quantity` is a real fixed-cost winner on the low-`K`, low-`p`
   Poisson family.
2. Its quality is not uniform across the whole fixed-cost grid.
3. The primary regression against the archived `L=4, p=4, K=5` winner is the `pipeline` state
   semantics drift introduced in `8b420dd`.
4. On top of that, the recent `2000`-episode recovery run still uses a weaker budget than the old
   trusted `5000`-episode canonical runs.
5. Backend drift exists and should be cleaned up, but it is not the main reason the recent
   `Q=50` rerun is underperforming.

## Local scale search

Once the normalization moved out of the env and into the policy, the next question became:

- is `Q`-style state scaling important only for the ordinal head,
- or is scale a first-order search parameter for the broader fixed-cost benchmark surface too?

The following local proxy sweeps answer that question on the same benchmark:

- problem: `lit_pois_mu5_l4_p4_k5`
- backend: Rust
- training: `200` CMA iterations
- evaluation: horizon `100000`, `3` seeds

Files:

- ordinal local sweep:
  `policy_search/studies/fixed_cost_ordinal_stability/scale_local_p50_results.md`
- depth-1 tree local sweep:
  `policy_search/studies/fixed_cost_ordinal_stability/scale_tree_d1_local_p50_results.md`
- depth-2 tree local sweep:
  `policy_search/studies/fixed_cost_ordinal_stability/scale_tree_d2_local_p50_results.md`
- combined comparison:
  `policy_search/studies/fixed_cost_ordinal_stability/scale_policy_comparison_local_p50_results.md`

### Cross-policy comparison at population 50

| Scale | Ordinal | Tree d1 | Tree d2 |
| ---: | ---: | ---: | ---: |
| 10 | 8.8424 | 8.7678 | 8.7771 |
| 15 | 9.5515 | 8.8061 | 8.7684 |
| 20 | 8.8334 | 8.7719 | 8.7729 |
| 25 | 8.8308 | 9.7522 | 8.7770 |
| 30 | 8.9003 | 8.7736 | 8.7779 |
| 40 | 9.0223 | 8.7738 | 9.7537 |
| 50 | 10.0741 | 9.7787 | 9.7795 |

### What the scale sweeps say

1. Scale is a first-order search parameter, not an ordinal-only quirk.

   Every tested policy family changes materially when the state is rescaled.

2. The good region is roughly in the `10` to `30` band.

   For the ordinal head, the best local proxy points are `20` and `25`.
   For the depth-2 tree, the whole `10` to `30` band is essentially flat and strong.

3. The old fixed-cost `50` scaling is not preferred at this short proxy budget.

   All three policies are worse at `50` than at the better points in the `10` to `30` range.

4. The sensitivity shape is policy-family specific.

   The ordinal head degrades fairly smoothly as the scale grows beyond `30`.
   The depth-2 tree is robust up to `30` and then drops sharply at `40`.
   The depth-1 tree shows isolated bad spikes, which suggests local-basin effects rather than a
   perfectly smooth monotone curve.

5. The depth-2 tree is the most robust family in this local proxy search.

   On this benchmark and budget, it stays in the `8.77` band from `10` through `30`.

These local sweeps do not replace the long canonical evaluations, but they are already decisive
about search geometry:

- scaling matters,
- the relevant good region is problem- and policy-family dependent,
- and hardcoding one old scale into the env is the wrong abstraction boundary.

## Next decisive experiments

The shortest path to a clean answer is:

1. promote the best local scale candidates to the long canonical budget:
   - ordinal: `20`, `25`, maybe `30`
   - tree depth-2: `10`, `20`, `25`, `30`
   - tree depth-1: rerun `25` and `50` once more to confirm whether those spikes are real or just
     single-seed basin artifacts
2. rerun the old canonical `L=4, p=4, K=5` ordinal policy on current Rust with the exact promoted
   protocol and policy-side scaling:
   - `training_episodes=5000`
   - `training_horizon=2000`
   - explicit `state_scale`
   - same evaluation setup
3. rerun the trusted tree policies under the same promoted protocol with the winning scale band
4. rerun the same exact protocol on current Python
4. compare:
   - if the promoted reruns recover the `~8.77` basin, the remaining issue was mainly search
     geometry and protocol mismatch
   - if Python recovers and Rust does not, the remaining gap is a backend implementation issue

## Reproducible helper

Use the helper scripts in this folder to reproduce the analysis:

- `summarize_fixed_cost_ordinal_history.py`
- `ablate_state_drift.py`
- `replay_exact_ordinal.py`
- `collect_policy_scale_results.py`

`summarize_fixed_cost_ordinal_history.py` reads benchmark result JSONs and prints a compact table.

`ablate_state_drift.py` evaluates the archived good ordinal checkpoint under the current env,
old-scale emulation, old-init emulation, and the fully old env to isolate which semantic change
caused the regression.

`replay_exact_ordinal.py` is now the generic fixed-cost replay helper for the single-policy scale
experiments in this note.

`collect_policy_scale_results.py` turns completed scale-sweep result directories into markdown
tables for a chosen policy family.
