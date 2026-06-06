# OWMR second PPO-gap AutoResearch, instance 13

Task focus: choose a second one-warehouse multi-retailer case that currently beats the in-repo
gate but does not beat the published Kaynov PPO reference, then design a policy direction that can
move toward PPO.

Chosen target: `kaynov2024_instance_13`.

- published PPO cost: `79727.39` (Kaynov et al. 2024 Table A.3, table-only in this repo)
- prior best repo learned cost: `85974.7852 +/- 88.2862`
- prior in-repo gate: `91890.2542 +/- 99.5618`
- prior learned vs PPO: `-7.8359%`

Instance 13 is a better second target than instance 14 for bounded AutoResearch because it already
has a strong learned policy that beats the tuned in-repo gate beyond paired SEM, while instance 14
currently ties or loses to the gate. The remaining gap is therefore a policy-class gap against PPO,
not a failure to clear the in-repo base-stock gate.

## Runner addition

The runner can now embed a `symmetric_echelon_targets` linear checkpoint into per-retailer target
modes:

- `echelon_targets`: `(W, shared R)` -> `(W, R_1, ..., R_K)`
- `echelon_targets_with_alloc_targets`: `(W, shared R)` -> `(W, R_1, ..., R_K, A_1, ..., A_K)`

The embedding is not a raw parameter copy. `symmetric_echelon_targets` uses discrete-grid bounds
with nonzero lower action values; `echelon_targets` uses `vector_quantity` with zero lower bounds.
For linear leaves, the runner adds the old action minimum to the copied leaf biases so
`min_old + softplus(raw_old)` is preserved as `min_new + softplus(raw_new)` in the large-raw regime.
A tiny smoke confirmed that the converted model no longer collapses to the gate.

## Per-retailer target restart

Command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_13 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_13_symmetric_echelon_targets_linear_pop32_396_600/model_params.npy \
  --sigma_init 0.5 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation proportional \
  --same_seed \
  --seed 734 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_13_echelon_targets_from_symmetric_restart_sigma0.5_seed734.json
```

Result:

- learned `84469.4978 +/- 88.8043`, deployed `trained_xbest`, evaluated under proportional
- initializer `85197.1963`
- gate `91890.2542 +/- 99.5618`
- paired gate - learned `+7420.7563 +/- 53.9795`
- published PPO `79727.39`
- learned vs PPO `-5.9479%`

This closes the PPO gap from `-7.8359%` to `-5.9479%`, but does not beat PPO.

## Local refinement

Command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_13 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_13_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen200_batch16_proportional_crn_sig0p5_seed734_1692_200/model_params.npy \
  --sigma_init 0.25 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation proportional \
  --same_seed \
  --seed 735 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_13_echelon_targets_restart_from_sigma0.5_sigma0.25_seed735.json
```

Result:

- learned `84399.2900 +/- 89.5883`, deployed `trained_xbest`, evaluated under proportional
- initializer `84469.4978`
- gate `91890.2542 +/- 99.5618`
- paired gate - learned `+7490.9641 +/- 53.3485`
- published PPO `79727.39`
- learned vs PPO `-5.8598%`

This is the current best repo-native result for `kaynov2024_instance_13`. It improves the prior
symmetric learned policy by `1575.4951` cost units and improves the first per-retailer restart by
`70.2078` cost units. The remaining PPO gap is `4671.9000` cost units.

## Negative / limiting evidence

Absolute-state augmentation from the new per-retailer checkpoint did not help at `sigma_init=0.125`.
The trained `xbest` overfit badly, and the honest deployment guard fell back to the initializer.

Command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_13 \
  --budget full \
  --policy_action_mode echelon_targets \
  --policy_state_mode absolute_augmented \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_13_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen200_batch16_proportional_crn_sig0p25_seed735_1692_200/model_params.npy \
  --sigma_init 0.125 \
  --gate_search_paths 64 \
  --training_episodes 160 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation proportional \
  --same_seed \
  --seed 736 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_13_echelon_targets_absolute_augmented_restart_from_sigma0.25_sigma0.125_seed736.json
```

Result:

- learned `84399.2900 +/- 89.5883`, deployed `init_params_anchor`
- trained `xbest` held-out cost `117694.1946`
- published PPO `79727.39`
- learned vs PPO `-5.8598%`

This argues that the augmented-state neighborhood needs either a much smaller sigma, a staged
training schedule, or explicit regularization/distillation. As run, it is not a useful PPO-gap path.

## Interpretation

The productive policy-design move for the second case was expanding the symmetric target policy
into per-retailer target outputs while preserving the incumbent. This gives CMA-ES a symmetry-
breaking direction and produces a real multi-SEM gate improvement, but it still does not reach PPO.

Next bounded directions:

- continue per-retailer restarts from the `84399.2900` checkpoint with smaller sigma values;
- try `echelon_targets_with_alloc_targets` from the same checkpoint so rationing priorities can
  differ from replenishment targets;
- retry absolute-state augmentation only with a much smaller sigma or fixed-path distillation;
- investigate whether PPO's advantage comes from non-base-stock action timing rather than target
  asymmetry alone.
