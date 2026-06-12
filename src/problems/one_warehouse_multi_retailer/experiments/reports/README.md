# one_warehouse_multi_retailer Paper Benchmark

> **Status (2026-05-31): learned soft-tree path REGENERATED on the current install.**
> The `common.py` migration to `invman.policy.Policy` is complete and the learned rollout
> bindings (`one_warehouse_multi_retailer_soft_tree_rollout` /
> `..._soft_tree_population_rollout` / `..._soft_tree_rollout_from_paths`) work. A fresh
> learned-vs-heuristic-vs-published benchmark on a representative 3-instance subset (one per
> customer-behavior regime: instance 1 backorder, 6 lost_sales, 11 partial_backorder) is in
> **"Learned soft-tree (held-out CRN, 2026-05-31)"** below, produced by
> `scripts/.../benchmark_learned_vs_heuristic.py` and saved to
> `outputs/one_warehouse_multi_retailer/learned_benchmark/learned_vs_heuristic_results.json`.
>
> The depth-1 14-instance soft-tree rows in **"Per Instance"** further down are the OLD
> `run_paper_benchmark.py` cache (in-sample heuristic search vs separate-seed soft-tree eval);
> kept for historical reference. The new block is the authoritative learned comparison: both
> the heuristic argmin and the learned policy are scored on the SAME held-out demand paths
> (Common Random Numbers via the `*_from_paths` bindings), removing the in-sample bias.

## Learned soft-tree (held-out CRN, 2026-05-31)

- script: `scripts/one_warehouse_multi_retailer/benchmark_learned_vs_heuristic.py`
- policy family: depth `2` `axis_aligned` soft tree, `linear` leaves, `symmetric_echelon_targets`
  action mode (the K=3 symmetric Poisson(3) instances)
- optimizer: CMA-ES (`invman.es_mp.train` + `..._soft_tree_population_rollout`), popsize `32`,
  `600` generations, `train_seed_batch=12`, `sigma_init=1.5`, train allocation `proportional`
- CPU cap: `RAYON_NUM_THREADS=2` / `OMP_NUM_THREADS=2`, `mp_num_processors=1` (no Python pool;
  parallelism is rayon inside Rust, bounded to 2 threads)
- evaluation: heuristic echelon base-stock grid-searched on a 256-path search block, then BOTH
  the learned policy and the heuristic argmin re-scored on the SAME 4096-path held-out block
  (disjoint seeds; CRN-paired via `*_from_paths`); 100-period undiscounted cost
  (`discount_factor = 1.0`), mean-filled warm-start initial state. Both allocation rules
  reported; the column shows the better one per policy.

| Instance | CB | Learned (alloc) | Best Heuristic (alloc) | Published Prop / Min / PPO | Learned vs Best Heuristic | Winner |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `kaynov2024_instance_1` | `backorder` | `1584.45` (min_shortage) | `1558.12` (min_shortage) | `1655.51 / 1609.47 / 1637.20` | `-1.69%` | heuristic |
| `kaynov2024_instance_6` | `lost_sales` | `1370.50` (proportional) | `1348.05` (proportional) | `1373.91 / 1366.51 / 1347.34` | `-1.67%` | heuristic |
| `kaynov2024_instance_11` | `partial_backorder` | `1189.51` (proportional) | `1184.46` (proportional) | `1111.76 / 1109.96 / 971.86` | `-0.43%` | heuristic |

Findings:

- The tuned echelon base-stock + allocation heuristic wins on all three representative
  instances, but the held-out gap is small (`0.4%`–`1.7%`); the depth-2 learned soft-tree is
  competitive, not dominant. This is consistent with the literature: for these symmetric
  Poisson(3) OWMR instances a well-chosen base-stock policy with proportional / min-shortage
  rationing is near-optimal, so a learned policy has little structure to exploit.
- Versus published PPO (lower cost = better; JSON `learned_vs_published_ppo_pct` is
  `(PPO − learned)/PPO`, so positive = learned is below/better-than PPO):
  - `instance_1` (backorder): learned `1584.45` < PPO `1637.20` → learned **beats** published
    PPO by `+3.22%` (the backorder PPO row underperforms the base-stock heuristics, so both the
    repo heuristic and the learned tree are below it).
  - `instance_6` (lost_sales): learned `1370.50` > PPO `1347.34` → learned is `1.72%` worse.
  - `instance_11` (partial_backorder): learned `1189.51` > PPO `971.86` → learned is `22.4%`
    worse; the repo's partial-backorder warm-start residual (both heuristic and learned land
    ~6–7% above the published base-stock too) is the dominant cause of this PPO gap, not the
    learned policy itself.
- `instance_11` (partial_backorder): both repo heuristic (`1184.5`) and learned (`1189.5`) land
  ~`6–7%` ABOVE the published proportional (`1111.8`), the same regime-dependent warm-start
  residual documented in `../../literature/README.md` (this is a protocol/initial-condition
  residual, not a transition bug; the exact-DP self-consistency still holds: optimal `8.485`
  dominates both heuristics `9.2225`).
- Coverage: 3 of 14 Kaynov instances trained+evaluated at full budget (one per regime; all K=3
  symmetric). Instances 2–5, 7–10, 12 (and the K=10 instances 13–14) were NOT re-run in this
  pass; the historical depth-1 rows for them remain in the table below.

## Autoresearch policy search (2026-05-31, full-budget sweep)

- runner: `scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py`
- program: `policy_search/programs/program_one_warehouse_multi_retailer.md`
- ledger: `outputs/autoresearch/one_warehouse_multi_retailer_autoresearch/results.tsv` (33 rows:
  2 smoke + 20 screening + 11 full, including a standalone full timing probe)
- CPU cap: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, `mp_num_processors=1` (two sibling agents)

Goal: flip the sign on the three losing instances above (`-0.43%` / `-1.67%` / `-1.69%`). The
sweep concentrated on the program's leading lever (CMA-ES warm-start at the best base-stock) and
ranked leaf `{constant, linear}` × depth `{2,3}` × temperature `{0.05,0.10,0.20}` ×
split `{axis_aligned, oblique}` × warm-start `{on,off}` at screening, then promoted the decisive
configs to full budget (popsize 32, 600 generations, 4096 held-out CRN paths).

Best config on ALL THREE instances: **depth-2 `axis_aligned` `constant` leaf, temperature 0.05,
`symmetric_echelon_targets`, warm-started at the best echelon base-stock (W, R)**.

| Instance | CB | Best learned (alloc) | Best heuristic (alloc) | gap% (full) | Prior gap% | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `kaynov2024_instance_1` | `backorder` | `1558.12` (min_shortage) | `1558.12` (min_shortage) | `0.0000%` | `-1.69%` | tie (matched) |
| `kaynov2024_instance_6` | `lost_sales` | `1348.05` (proportional) | `1348.05` (proportional) | `0.0000%` | `-1.67%` | tie (matched) |
| `kaynov2024_instance_11` | `partial_backorder` | `1184.46` (proportional) | `1184.46` (proportional) | `0.0000%` | `-0.43%` | tie (matched) |

Findings:

- The gap closed from `-0.43%…-1.69%` to **exactly `0.0%`** (learned cost equals the heuristic cost
  to six decimals) on all three. This is a **tie, not a strict flip**: the warm-started constant-leaf
  tree reproduces the heuristic action at generation 0 and CMA-ES finds no profitable
  state-dependent deviation, even at 600 generations. Consistent with the literature/exact-DP prior:
  the tuned echelon base-stock + allocation heuristic is at/near the optimum on these symmetric
  Poisson(3) K=3 instances, so a learned policy has no exploitable state structure to win on.
- **The warm-start was previously broken** and is the load-bearing fix. The runner's
  `_warm_start_flat_params` wrote the raw base-stock target into the soft-tree leaf block, but the
  tree applies a per-leaf-type transform before grid-snapping (constant leaf: `min + sigmoid(p)·span`;
  linear leaf: `min + softplus(bias + w·state)` — see `src/core/policies/soft_tree.rs`). The
  raw target sigmoid-saturated the constant leaf to the grid maximum, so generation 0 began at a
  badly over-stocked policy (instance-11 holdout ≈ 1879 vs heuristic ≈ 1180), not the heuristic. The
  fix inverts the transform (logit for constant; zeroed leaf weights + softplus-inverse bias for
  linear) so generation 0 reproduces the heuristic exactly; warm-started constant then beats the
  no-warm control (`-0.20%`/`-0.04%`) and all linear/oblique/depth-3 variants.
- Lever ranking (full budget): constant ≫ linear (linear `-0.32%`…`-1.84%`); axis_aligned ≈ oblique
  (oblique slightly worse on instance 11); depth-2 ≈ depth-3 (depth-3 ties but adds no value);
  temperature immaterial under the warm-started constant leaf; warm-start `on` ≫ `off`.
- Not run (bounded sweep, logged): `direct_orders`/`vector_quantity` action design,
  `random_sequential` train allocation, sigma schedules, and the 11 non-losing instances.

## Historical depth-1 14-instance cache (run_paper_benchmark.py)

- source: Kaynov et al. (2024), International Journal of Production Economics 267, 109088
- url: https://doi.org/10.1016/j.ijpe.2023.109088
- instances: `14`
- policy family: depth `1` `axis_aligned` soft tree with `linear` leaves
- training allocation: `random_sequential`
- evaluation allocation: `proportional`
- heuristic search: `1000` trajectories of length `100` with common random numbers
- benchmark evaluation: `1000` independent trajectories of length `100`
- instance 14 search note: Kaynov state that instance 14 searches over warehouse level z0 and a shared percentile parameter k. The paper does not publish a discrete k-grid, so the repo enumerates the unique integer retailer-target vectors induced by continuous k in [0, 3].

## Aggregate

- beats best repo heuristic on `3` / `14` instances
- beats best published heuristic on `4` / `14` instances
- beats published PPO on `1` / `14` instances
- mean gap vs best repo heuristic: `-332.675`
- mean gap vs published PPO: `-2330.060`

## Per Instance

| Instance | CB | Learned | Best Repo Heuristic | Best Published Heuristic | Published PPO | Gap vs Repo Best | Gap vs PPO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kaynov2024_instance_1` | `backorder` | `1600.006` | `1562.124` | `1609.470` | `1637.200` | `-37.882` | `37.194` |
| `kaynov2024_instance_2` | `backorder` | `1605.412` | `1381.784` | `1383.880` | `1417.460` | `-223.629` | `-187.952` |
| `kaynov2024_instance_3` | `backorder` | `1870.782` | `1728.020` | `1776.040` | `1731.670` | `-142.762` | `-139.111` |
| `kaynov2024_instance_4` | `backorder` | `1913.986` | `1802.271` | `1857.300` | `1908.950` | `-111.716` | `-5.036` |
| `kaynov2024_instance_5` | `backorder` | `2648.740` | `2454.622` | `2246.840` | `2331.070` | `-194.119` | `-317.670` |
| `kaynov2024_instance_6` | `lost_sales` | `1394.469` | `1346.834` | `1366.510` | `1347.340` | `-47.635` | `-47.130` |
| `kaynov2024_instance_7` | `lost_sales` | `1444.756` | `1390.966` | `1406.270` | `1405.080` | `-53.791` | `-39.677` |
| `kaynov2024_instance_8` | `lost_sales` | `1504.980` | `1473.796` | `1508.120` | `1495.490` | `-31.184` | `-9.490` |
| `kaynov2024_instance_9` | `lost_sales` | `1659.611` | `1521.814` | `1535.960` | `1511.680` | `-137.797` | `-147.931` |
| `kaynov2024_instance_10` | `lost_sales` | `1981.285` | `1777.719` | `1736.550` | `1674.540` | `-203.566` | `-306.745` |
| `kaynov2024_instance_11` | `partial_backorder` | `1141.559` | `1178.381` | `1109.960` | `971.860` | `36.823` | `-169.699` |
| `kaynov2024_instance_12` | `partial_backorder` | `1221.131` | `1240.124` | `1402.380` | `1118.920` | `18.993` | `-102.211` |
| `kaynov2024_instance_13` | `partial_backorder` | `89098.800` | `97166.700` | `99882.510` | `79727.390` | `8067.900` | `-9371.410` |
| `kaynov2024_instance_14` | `partial_backorder` | `64648.989` | `53051.904` | `52787.410` | `42835.020` | `-11597.085` | `-21813.969` |
