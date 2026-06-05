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

When `--init_params_npy` is used, deployment selects the best held-out candidate among trained
`xbest`, the loaded initializer, and the gate anchor.

## Best new result

The best policy found in this follow-up is a two-step small-sigma restart:

1. restart the old proportional incumbent with `train_allocation=min_shortage`, `sigma_init=0.10`,
   `pop24 x 300`, producing `1141.0050`;
2. restart that new model with `sigma_init=0.05`, again under `train_allocation=min_shortage`,
   producing `1140.9575`.

Best command:

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/one_warehouse_multi_retailer/run_asymmetric_learned_vs_gate.py \
  --reference kaynov2024_instance_12 \
  --budget full \
  --policy_action_mode echelon_targets \
  --leaf_type linear \
  --warm_start_at_best_base_stock \
  --init_params_npy outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen300_batch8_min_shortage_sig0p1_seed700_228_300/model_params.npy \
  --sigma_init 0.05 \
  --gate_search_paths 64 \
  --training_episodes 300 \
  --es_population 24 \
  --train_seed_batch 8 \
  --holdout_paths 4096 \
  --train_allocation min_shortage \
  --seed 702 \
  --output_json outputs/one_warehouse_multi_retailer/asymmetric_learned/kaynov2024_instance_12_echelon_targets_linear_restart_from_sigma0.10_sigma0.05_seed702.json
```

Result:

- learned `1140.9575 +/- 2.1915`, deployed `trained_xbest`, evaluated under proportional
- gate `1169.5905 +/- 2.0548`
- paired gate - learned `+28.6329 +/- 0.9554`
- gap vs gate `+2.4481%`
- published PPO `1118.92`
- gap vs PPO `-1.9695%`

This improves the prior best (`1154.0874`) by `13.1299` cost units and closes the PPO gap from
`3.1430%` to `1.9695%`, but does not beat PPO.

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
4096-path held-out block (`+28.63 +/- 0.96` vs gate), but still leaves `22.04` cost units to
published PPO.

Next high-yield experiment: continue from the new `1140.9575` model with smaller local restarts,
not from the older proportional incumbent:

- start from `outputs/one_warehouse_multi_retailer/asymmetric_learned/models/asym_kaynov2024_instance_12_echelon_targets_linear_d2_axis_aligned_t0.1_pop24_gen300_batch8_min_shortage_sig0p05_seed702_228_300/model_params.npy`
- keep `train_allocation=min_shortage`
- test `sigma_init` in `{0.02, 0.05, 0.10}` with `pop24 x 300`, `train_seed_batch=8`
- use 4096 held-out paths and select by paired improvement beyond SEM
- only promote a restart if it beats `1140.9575`; longer runs have already shown regression risk
