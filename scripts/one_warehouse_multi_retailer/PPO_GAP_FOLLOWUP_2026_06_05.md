# OWMR PPO-gap follow-up, instance 12

Task focus: new levers for the case that already beats the tuned in-repo gate but trails the
published Kaynov PPO. The closest row is `kaynov2024_instance_12`:

- prior best learned: `1154.0874` (`echelon_targets`, linear, trained under proportional)
- tuned in-repo gate: `1169.5905`
- published PPO: `1118.92`
- prior gap to PPO: `-3.1430%` (`(ppo - learned) / ppo`)

All runs below used held-out CRN evaluation against the same gate protocol. Outputs are under
`outputs/one_warehouse_multi_retailer/asymmetric_learned/` (gitignored); this note preserves the
exact commands and metrics.

## Runner additions

`run_asymmetric_learned_vs_gate.py` now exposes bounded search knobs:

- `--depth`, `--temperature`, `--split_type`
- `--training_episodes`, `--es_population`, `--train_seed_batch`, `--holdout_paths`
- `--init_params_npy` for CMA-ES restart/refinement from a saved policy
- `--direct_order_gate_init`, a near-gate raw-order initializer for `direct_orders` + linear leaf
- `--same_seed` to use common random numbers within each CMA-ES population batch
- `--train_on_fixed_paths` to score each CMA-ES population on the same explicit
  demand-path block, using `train_seed_batch` as the number of fixed training paths
- `published_ppo_cost` and `learned_vs_published_ppo_pct` JSON aliases for simple PPO-gap scans
- `trained_model_params_npy` in JSON so promoted policies can be resumed without reconstructing paths
- `echelon_targets_with_alloc_targets`, a decoupled target mode with separate retailer order
  targets and retailer allocation targets
- automatic embedding of an old `echelon_targets` checkpoint into the decoupled mode by copying
  retailer target outputs into both target blocks
- `--policy_state_mode absolute_augmented`, which keeps the normalized state features and appends
  the per-state scale, raw total echelon inventory position, and raw retailer inventory positions
- checkpoint embedding from normalized-state soft-tree params into augmented-state params by zeroing
  the newly added input-feature weights

When `--init_params_npy` is used, deployment selects the best held-out candidate among trained
`xbest`, the loaded initializer, and the gate anchor.

## Best same-mode restart result

The best policy found before adding a new action geometry was a small-sigma restart chain followed
by lower-noise CRN refinement:

1. restart the old proportional incumbent with `train_allocation=min_shortage`, `sigma_init=0.10`,
   `pop24 x 300`, producing `1141.0050`;
2. restart that new model with `sigma_init=0.05`, again under `train_allocation=min_shortage`,
   producing `1140.9575`;
3. restart that model with `sigma_init=0.02`, again under `train_allocation=min_shortage`,
   producing `1140.8092`;
4. restart that model with `sigma_init=0.01`, again under `train_allocation=min_shortage`,
   producing `1140.3179`;
5. restart that model with `sigma_init=0.005`, `--same_seed`, and `train_seed_batch=16`,
   producing `1140.1962`;
6. restart that model with `sigma_init=0.0025`, `--same_seed`, and `train_seed_batch=16`,
   producing `1140.0686`;
7. restart that model with `sigma_init=0.00125`, `--same_seed`, and `train_seed_batch=16`,
   producing `1140.0393`.

Best command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen200_batch16_min_shortage_crn_sig0p0025_seed717_228_200/model_params.npy \
  --sigma_init 0.00125 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --same_seed \
  --seed 718 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_from_crn_sigma0.0025_sigma0.00125_seed718.json
```

Result:

- learned `1140.0393 +/- 2.1746`, deployed `trained_xbest`, evaluated under proportional
- gate `1169.5905 +/- 2.0548`
- paired gate - learned `+29.5511 +/- 0.9431`
- gap vs gate `+2.5266%`
- published PPO `1118.92`
- gap vs PPO `-1.8875%`

This improves the prior best (`1154.0874`) by `14.0481` cost units and closes the PPO gap from
`3.1430%` to `1.8875%`, but does not beat PPO. It is now superseded by the decoupled
allocation-target mode below.

## Decoupled allocation-target result

Implemented `echelon_targets_with_alloc_targets` as the first richer policy class after the
same-mode restart chain saturated.

- action dimension: `1 + 2K` controls: warehouse order-up-to target, K retailer order targets,
  and K retailer allocation targets
- order logic: use the first `K+1` controls with `echelon_base_stock_orders`
- rationing logic: pass the second K retailer controls as `retailer_target_inventory_positions`
  for `min_shortage`
- warm start: initialize order targets and allocation targets both from the incumbent/gate retailer
  targets so the old policy remains representable
- checkpoint expansion: load an old K+1 `echelon_targets` soft-tree checkpoint into the new 1+2K
  model by copying retailer outputs into both retailer target blocks

Two full CRN restarts from the `1140.0393` incumbent improved the frontier:

- `sigma_init=0.005`, seed `719`: learned `1139.9194 +/- 2.1705`, paired gate - learned
  `+29.6710 +/- 0.9455`, gap vs PPO `-1.8768%`.
- `sigma_init=0.0025`, seed `720`: learned `1139.8884 +/- 2.1677`, paired gate - learned
  `+29.7020 +/- 0.9443`, gap vs PPO `-1.8740%`.

Best command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_with_alloc_targets_linear_d2_axis_aligned_t0.1_pop24_gen200_batch16_min_shortage_crn_sig0p005_seed719_372_200/model_params.npy \
  --sigma_init 0.0025 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --same_seed \
  --seed 720 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_with_alloc_targets_restart_from_sigma0.005_sigma0.0025_seed720.json
```

Result:

- learned `1139.8884 +/- 2.1677`, deployed `trained_xbest`, evaluated under proportional
- gate `1169.5905 +/- 2.0548`
- paired gate - learned `+29.7020 +/- 0.9443`
- gap vs gate `+2.5395%`
- published PPO `1118.92`
- gap vs PPO `-1.8740%`

This improved the saturated same-mode checkpoint by only `0.0309` cost units and still left
`20.9684` cost units to the published PPO number. It is now superseded by the augmented-state
result below.

## Absolute-state augmented result

The first follow-up from the decoupled target plateau was to expose absolute magnitude to the tree.
The default policy state still uses the old normalized layout, but `--policy_state_mode
absolute_augmented` appends:

- the normalization scale used by the normalized features,
- raw total system echelon inventory position,
- raw retailer inventory positions.

The old normalized `372`-parameter checkpoint is embedded into the `527`-parameter augmented model
by copying all old split/leaf weights into the first normalized feature columns and setting weights
for the new absolute features to zero. This preserves the incumbent before CMA-ES starts.

Best command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets \
  --policy_state_mode absolute_augmented \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_with_alloc_targets_linear_d2_axis_aligned_t0.1_pop24_gen200_batch16_min_shortage_crn_sig0p0025_seed720_372_200/model_params.npy \
  --sigma_init 0.00125 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --same_seed \
  --seed 721 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_with_alloc_targets_absolute_augmented_restart_from_normalized_sigma0.00125_seed721.json
```

Result:

- learned `1139.5526 +/- 2.1648`, deployed `trained_xbest`, evaluated under proportional
- gate `1169.5905 +/- 2.0548`
- paired gate - learned `+30.0378 +/- 0.9453`
- gap vs gate `+2.5682%`
- published PPO `1118.92`
- gap vs PPO `-1.8440%`

This is the current best repo-native result for `kaynov2024_instance_12`. It improves the
decoupled normalized-state checkpoint by `0.3358` cost units and still leaves `20.6326` cost units
to the published PPO number.

## Fixed-path objective result

The next bounded lever was to train each CMA-ES population on a fixed explicit CRN path block
instead of resampling by seed inside the population evaluator. This reduces within-generation
ranking noise but can overfit the small training block, so the headline still comes from the same
4096-path held-out CRN block.

Command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets_with_alloc_targets \
  --policy_state_mode absolute_augmented \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_with_alloc_targets_linear_d2_axis_aligned_t0.1_absolute_augmented_pop24_gen200_batch16_min_shortage_crn_sig0p00125_seed721_527_200/model_params.npy \
  --sigma_init 0.001 \
  --gate_search_paths 64 \
  --training_episodes 200 \
  --es_population 24 \
  --train_seed_batch 16 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --train_on_fixed_paths \
  --seed 722 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_with_alloc_targets_absolute_augmented_fixedpaths_restart_sigma0.001_seed722.json
```

Result:

- learned `1139.5526 +/- 2.1648`, deployed `init_params_anchor`, evaluated under proportional
- trained `xbest` held-out cost `1140.9027`, so deployment fell back to the loaded incumbent
- gate `1169.5905 +/- 2.0548`
- paired gate - learned `+30.0378 +/- 0.9453`
- published PPO `1118.92`
- gap vs PPO `-1.8440%`
- fixed training paths: `16`, demand seed start `600000`, allocation seed `750000`

This did not improve the frontier. It is useful limiting evidence that fixed-path ranking alone is
not enough; with only 16 fixed training paths it improved the training objective but overfit relative
to the held-out block.

## Negative / limiting evidence

Restart from prior best, proportional training, same 300 x pop24 budget:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets \
  --leaf_type linear --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_pop32_228_600/model_params.npy \
  --sigma_init 0.35 --gate_search_paths 64 --training_episodes 300 \
  --es_population 24 --train_seed_batch 8 --holdout_paths 4096 \
  --train_allocation proportional --seed 456 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_sigma0.35_mid_seed456.json
```

Held-out trained `xbest` was worse (`1172.8318`), so deployment fell back to the loaded incumbent
`1154.0874`.

Full-budget promotion of the min-shortage restart:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets \
  --leaf_type linear --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_pop32_228_600/model_params.npy \
  --sigma_init 0.35 --gate_search_paths 64 --train_allocation min_shortage --seed 654 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_sigma0.35_minshort_full_seed654.json
```

Held-out trained `xbest` was worse (`1159.2170`), so deployment fell back to the loaded incumbent
`1154.0874`. Longer pop32 x 600 training did not preserve the 300-generation improvement.

Restart from the new `1150.1581` min-shortage model with `sigma_init=0.20`:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets \
  --leaf_type linear --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_228_300/model_params.npy \
  --sigma_init 0.20 --gate_search_paths 64 --training_episodes 300 \
  --es_population 24 --train_seed_batch 8 --holdout_paths 4096 \
  --train_allocation min_shortage --seed 701 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_from_mid_sigma0.20_seed701.json
```

Held-out trained `xbest` was worse (`1151.3395`), so deployment fell back to the loaded incumbent
`1150.1581`.

Restart from the current `1140.8092` model with a larger `sigma_init=0.10`:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen300_batch8_min_shortage_sig0p02_seed703_228_300/model_params.npy \
  --sigma_init 0.10 \
  --gate_search_paths 64 \
  --training_episodes 300 \
  --es_population 24 \
  --train_seed_batch 8 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --seed 704 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_from_sigma0.02_sigma0.10_seed704.json
```

Held-out trained `xbest` was worse (`1148.2242`), so deployment fell back to the loaded incumbent
`1140.8092`. This argues for local restarts at or below `sigma_init=0.05`, or for a richer policy
class, rather than widening the CMA-ES neighborhood around the current incumbent.

Restart from the current `1140.3179` model with another `sigma_init=0.01` run:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen300_batch8_min_shortage_sig0p01_seed705_228_300/model_params.npy \
  --sigma_init 0.01 \
  --gate_search_paths 64 \
  --training_episodes 300 \
  --es_population 24 \
  --train_seed_batch 8 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --seed 706 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_from_sigma0.01_sigma0.01_seed706.json
```

Held-out trained `xbest` was worse (`1140.7797`), so deployment fell back to the loaded incumbent
`1140.3179`. The frontier is still moving, but improvements are now sub-unit and training noise is
large enough that CRN training (`--same_seed`) and smaller sigma values are the next bounded levers.

Two lower-noise CRN refinements from the `1140.1962` model still improved, but the gains are now
small:

- `sigma_init=0.0025`, `--same_seed`, `train_seed_batch=16`, seed `717`: trained `xbest`
  improved to `1140.0686`.
- `sigma_init=0.00125`, `--same_seed`, `train_seed_batch=16`, seed `718`: trained `xbest`
  improved to `1140.0393`.

The second run gained only `0.0293` cost units over the previous checkpoint, while the remaining
PPO gap is still `21.12` cost units. This is the clearest evidence so far that the current
`echelon_targets` depth-2 local neighborhood is close to saturated.

Random-sequential training from the gate anchor:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget full --policy_action_mode echelon_targets \
  --leaf_type linear --warm_start_at_best_base_stock --gate_search_paths 64 \
  --training_episodes 300 --es_population 24 --train_seed_batch 8 --holdout_paths 2048 \
  --train_allocation random_sequential --seed 123 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_d2_axis_t0.10_randseq_mid_seed123.json
```

Trained `xbest` was `1210.8718`; deployment fell back to gate anchor `1170.4443`.

Direct raw-order near-gate warm start:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 --budget screening --policy_action_mode direct_orders \
  --leaf_type linear --direct_order_gate_init --gate_search_paths 64 \
  --training_episodes 120 --holdout_paths 1024 --train_allocation proportional --seed 123 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_direct_orders_linear_neargate_screen120_seed123.json
```

The analytic near-gate direct initializer itself was very poor (`6044.0820`), and the trained
screen still lost badly: learned `1292.1348` vs gate `1165.3330` (`-10.8812%`). This rules out the
simple direct-order affine warm start as a useful path.

## Interpretation

The useful new lever is not raw direct orders or random-sequential training. It is a small-sigma
CMA-ES restart chain from the incumbent `echelon_targets` policy while training under
`min_shortage` and deploying under the better held-out allocation. The improvement is real on the
4096-path held-out block (`+29.70 +/- 0.94` vs gate), but still leaves `20.97` cost units to
published PPO.

Decoupling order targets from allocation targets is valid but only slightly useful under the old
normalized state. Exposing absolute magnitude is also valid and more useful than one more
same-mode restart, but the first augmented run still gained only `0.3358` cost units while the PPO
gap remains `20.6326` cost units. The next high-yield path should be residual/windowed targets,
learned allocation priorities, fixed-CRN objective training, or a PPO-style direct-order policy,
not only smaller local restarts of this depth-2 tree.
