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
- `published_ppo_cost` and `learned_vs_published_ppo_pct` JSON aliases for simple PPO-gap scans
- `trained_model_params_npy` in JSON so promoted policies can be resumed without reconstructing paths

When `--init_params_npy` is used, deployment selects the best held-out candidate among trained
`xbest`, the loaded initializer, and the gate anchor.

## Best new result

The best policy found in this follow-up is a small-sigma restart chain followed by one
lower-noise CRN refinement:

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
`3.1430%` to `1.8875%`, but does not beat PPO.

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
4096-path held-out block (`+29.55 +/- 0.94` vs gate), but still leaves `21.12` cost units to
published PPO.

Next high-yield experiment: stop relying on smaller local restarts alone. The current
`echelon_targets` action emits the same retailer target vector for ordering and for min-shortage
rationing. The Rust rollout already carries a `retailer_target_inventory_positions` hook, so the
next code-side candidate is a new action mode that emits separate order targets and allocation
targets.

- proposed mode: `echelon_targets_with_alloc_targets`
- action dimension: `1 + 2K` controls: warehouse order-up-to target, K retailer order targets,
  and K retailer allocation targets
- order logic: use the first `K+1` controls with `echelon_base_stock_orders`
- rationing logic: pass the second K retailer controls as `retailer_target_inventory_positions`
  for `min_shortage`
- warm start: initialize order targets and allocation targets both from the incumbent/gate retailer
  targets so the old policy remains representable
- first bounded screen: depth 2, linear leaves, `--same_seed`, `sigma_init=0.005`,
  `train_seed_batch=16`, 4096 held-out paths, promote only if it beats `1140.0393`
