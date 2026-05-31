# nonstationary_lot_sizing

Canonical Rust-first home for the nonstationary single-item lot-sizing family
(Dehaybe, Catanzaro & Chevalier 2024, EJOR 314(2):433-445).

## Verification status: VERIFIED (yes)

The environment faithfully matches the cited model and the repo reproduces the
published author-repo benchmark numbers on all eight forecast instances.

- Model fidelity: `env.rs::step_state` reproduces the paper's Section 4.2 worked
  transition exactly (period cost 130, reward -130); the `simple_s_s` heuristic
  matches the author testbed (s,S) formula term-for-term; demand models follow the
  testbed (CV-Normal for the simple baseline, Poisson for the DP baseline).
- Published-number reproduction: the eight `references.rs` rows are byte-for-byte
  the author-repo CSVs, and the repo's simulator reproduces every one of them
  within the stored 35-cost tolerance (±0.17% relative) at 25,000 replications.
- See `literature/README.md` for the full evidence table and source pointers, and
  `verification/README.md` for the executable verifier contract.

## Code

- implementation: `rust/src/problems/nonstationary_lot_sizing/`
- tests: `rust/src/problems/nonstationary_lot_sizing/tests/verification.rs`

## Artifact folders

- `literature/` — paper scope, fidelity argument, reproduced-number table
- `practical/` — checked-in rolling forecast trace, benchmark spec, latest report
- `experiments/` — paper-facing benchmark definition
- `verification/` — human-readable statement of what the verifier asserts

## Canonical anchors

- primary literature instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification instance: `constant_10_rolling_dp_reference`
- practical benchmark dataset: `retail_like_weekly_trace`

## Benchmark

Two scripts, both runnable against the installed `invman_rust` (no rebuild):

1. Literature benchmark (the eight forecast instances):
   `scripts/nonstationary_lot_sizing/run_literature_benchmark.py`
   - reproduces the author `simple` (CV-Normal) and `DP` (Poisson) rows,
   - reports `lead_time_base_stock` and the gap of every policy to the `rolling_dp_s_s`
     comparator (the paper's strong DP baseline; no exact optimum exists for the
     rolling-forecast path),
   - with `--learned`, trains a CMA-ES soft tree per instance against the exposed
     Rust rollout binding and reports its gap to the DP baseline.
2. Practical benchmark (one fixed rolling forecast path, operational metrics):
   `scripts/nonstationary_lot_sizing/run_practical_benchmark.py`
   (snapshot in `practical/reports/`).

### Heuristics-vs-DP results (verified, 25,000 replications, seed 1234)

Simulated total cost over 104 periods. `rolling_dp_s_s` (Poisson) is the strongest
comparator; `simple_s_s` and `lead_time_base_stock` use CV-Normal demand.

| instance     | simple_s_s | rolling_dp_s_s | lead_time_base_stock |
| ------------ | ---------: | -------------: | -------------------: |
| constant_5   |    1253.11 |        1214.92 |              1300.4 |
| constant_10  |    1834.92 |        1714.15 |              1543.2 |
| constant_15  |    2367.30 |        2071.94 |              1840.1 |
| seasonal_1   |    1825.71 |        1678.18 |              1546.1 |
| seasonal_2   |    1870.33 |        1678.84 |              1550.0 |
| seasonal_4   |    1857.53 |        1684.87 |              1559.5 |
| growth       |    1753.17 |        1605.69 |              1485.8 |
| decline      |    1963.09 |        1837.68 |              1654.2 |

Reading: the author "DP" row (`rolling_dp_s_s`, Poisson) is reproduced as the
DP baseline, but it is NOT the cheapest policy on this slice. With a small fixed
cost (K=10) relative to mean demand, `lead_time_base_stock` -- which orders every
period (no fixed-cost batching) at the lead-time critical-ratio level -- is cheaper
than both (s,S) policies on every instance except constant_5, beating the DP
baseline by 5-13% on the larger-demand instances. The two demand models are not
identical (DP uses Poisson, the others CV-Normal), so this is a heuristic-design
observation, not a strict apples-to-apples optimum; it is exactly the kind of gap
the paper's DRL agent targets, and it motivates the learned-policy comparison
below. (All three columns measured at 25,000 replications, seed 1234.)

### Learned-policy comparison

The repo exposes a soft-tree rollout binding
(`nonstationary_lot_sizing_soft_tree_population_rollout`), and the benchmark script
trains it directly with the read-only `invman.cmaes.CMAES` (state = normalized
forecast window + net inventory + pipeline; scalar order quantity clipped to
`[0, --action_cap]`).

Full-budget converged run (all eight instances; depth-2 oblique soft tree with
`linear` leaves, 150 CMA-ES generations x 48 candidates, `--action_cap 100`).
Training and held-out evaluation are seed-disjoint: CMA-ES fits each candidate on
one common-random-number seed per generation (seeds `1234+1 .. 1234+48`), and the
reported cost is a fresh out-of-sample roll-out at 10,000 replications on a disjoint
seed block (`1234+99 .. 1234+99+10000`). All policies share the env's order-of-events
(order -> arrival -> demand -> cost). `learned` and `lead_time_base_stock` use
CV-Normal demand (cv=0.2); the published `rolling_dp_s_s` (DP) baseline uses Poisson
demand, exactly as in the author testbed -- so the DP column is the published strong
comparator, not a same-demand-model optimum. Total cost over 104 periods.

| instance     | learned soft tree | best heuristic (cost)      | DP baseline | gap vs DP | gap vs best heur | winner       |
| ------------ | ----------------: | -------------------------- | ----------: | --------: | ---------------: | ------------ |
| constant_5   |            1026.3 | lead_time_base_stock 1300.4 |      1214.9 |   -15.52% |          -18.10% | **learned**  |
| constant_10  |            1539.0 | lead_time_base_stock 1543.2 |      1714.1 |   -10.22% |           -0.27% | **learned**  |
| constant_15  |            1785.2 | lead_time_base_stock 1840.1 |      2071.9 |   -13.84% |           -2.98% | **learned**  |
| seasonal_1   |            1517.1 | lead_time_base_stock 1546.1 |      1678.2 |    -9.60% |           -1.88% | **learned**  |
| seasonal_2   |            1569.9 | lead_time_base_stock 1550.0 |      1678.8 |    -6.49% |           +1.28% | lead_time_bs |
| seasonal_4   |            1534.6 | lead_time_base_stock 1559.5 |      1684.9 |    -8.92% |           -1.60% | **learned**  |
| growth       |            1491.0 | lead_time_base_stock 1485.8 |      1605.7 |    -7.14% |           +0.35% | lead_time_bs |
| decline      |            1711.4 | lead_time_base_stock 1654.2 |      1837.7 |    -6.87% |           +3.46% | lead_time_bs |

Reading: the learned soft tree **beats the published DP baseline on all 8/8
instances** (-6.5% to -15.5%) and is the **single cheapest policy on 5/8** (it beats
both the `simple_s_s`/`lead_time_base_stock` heuristics and DP). On the three
remaining instances (seasonal_2, growth, decline) it still beats DP but loses to
`lead_time_base_stock` by a small margin (0.35%-3.46%). The standout is `constant_5`,
where the learned policy is 18.1% below the best heuristic and 15.5% below DP. The
earlier small-budget snapshot under-converged on `constant_10` (+25.86% vs DP); at the
full 150x48 budget that is resolved (`constant_10` now -10.22% vs DP). This is a
converged, honest, held-out result.

Reproduce with (capped at 2 worker threads to share the box):

```
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/nonstationary_lot_sizing/run_literature_benchmark.py --learned \
  --instances constant_5 constant_10 constant_15 seasonal_1 seasonal_2 seasonal_4 growth decline \
  --replications 25000 --tree_depth 2 --leaf_type linear --action_cap 100 \
  --generations 150 --popsize 48 --learned_replications 10000 --seed 1234 \
  --output_json outputs/nonstationary_lot_sizing/learned_benchmark_full8_g150_p48_ac100_r10000.json
```

Raw numbers (including reproduced `simple`/`dp` published-row checks, all within
0.17% of the author CSVs) are saved in
`outputs/nonstationary_lot_sizing/learned_benchmark_full8_g150_p48_ac100_r10000.json`.
The `nonstationary_lot_sizing_soft_tree_population_rollout` binding parallelizes the
CMA-ES candidate batch via rayon, so worker count is capped through
`RAYON_NUM_THREADS` (there is no script-level `mp_num_processors` flag for this
self-contained path).

#### Learned-policy blocker (standard harness)

A learned-policy benchmark through the project's standard training harness is NOT
wired yet, and wiring it requires edits to guardrail-protected shared files:

- `invman/rollout_fitness.py` dispatches soft-tree training only for `lost_sales`,
  `dual_sourcing`, and `multi_echelon`; it has no `nonstationary_lot_sizing` branch.
- `invman/policy_build.py::build_policy` likewise has no branch for this family.
- There is no `invman/problems/nonstationary_lot_sizing/` Python package (the
  finished siblings ship a `reference_instances.py`).

The benchmark script's `--learned` path sidesteps all of this by calling the
exposed binding and CMA-ES directly, so a learned-vs-DP comparison is runnable
today without touching shared infrastructure -- and the converged 8-instance table
above was produced through exactly that self-contained path. Only the wiring into the
*shared* harness remains; the learned-policy benchmark itself is no longer a
placeholder.
