# Paper Benchmark

Paper objective for this family:

- design policy classes for forecast-driven lot sizing
- optimize their parameters with CMA-ES
- compare them against:
  - problem heuristics
  - the strongest benchmark baseline when no exact optimum is available for the reported path

## Reported Instances

Use two reported slices.

1. literature benchmark slice
   - `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
   - `dehaybe2024_lostsales_lt2_b5_k10_seasonal_2`
   - `dehaybe2024_lostsales_lt2_b5_k10_growth`
2. practical trace slice
   - `retail_like_weekly_trace`

Reason:

- the literature slice covers constant, seasonal, and trend demand structures
- the practical trace slice gives a fixed rolling-forecast path with operational metrics

## Learned Policy Families

Initial reported learned-policy family:

- `soft_tree`
  - depth `2`
  - leaf types:
    - `linear`
    - `sigmoid_linear`

The first paper benchmark should focus on one learned family so the comparison is about policy
design and CMA-ES fitting, not about a large architecture sweep.

## Heuristic Comparators

Report against:

- `lead_time_base_stock`
- `simple_s_s`
- `rolling_dp_s_s`

## Exact / Optimal Comparator

Not available as a full exact optimum for the reported forecast-path experiments.

Comparator rule:

- use `rolling_dp_s_s` as the strongest benchmark baseline
- do not present it as the global optimal policy for the nonstationary path problem

## Reported Metrics

Literature slice:

- simulated mean cost
- shortage rate
- gap to `rolling_dp_s_s`
- gap to best simple heuristic

Practical trace slice:

- mean period cost
- shortage rate
- cycle service level
- mean holding inventory
- mean order quantity

## Report Table Intent

The paper table for this family should show:

1. whether CMA-ES can design a policy that closes the gap to `rolling_dp_s_s`
2. whether the learned policy improves over simpler heuristics on seasonal and trend cases
3. how the learned policy behaves on a fixed operational forecast path

## Learned-Policy Results (CMA-ES soft tree, converged)

Run: `scripts/nonstationary_lot_sizing/run_literature_benchmark.py --learned` on all
eight forecast instances.

Budget / protocol:

- learned family: `soft_tree`, depth 2, oblique splits, `linear` leaves
- CMA-ES: 150 generations x 48 candidates, `sigma_init=0.5`, action clipped to [0, 100]
- objective: full 104-period rolling-forecast roll-out, CV-Normal demand (cv=0.2)
- training fitness: 1 common-random-number seed per candidate per generation
  (seeds `1235..1282`)
- held-out evaluation: fresh disjoint seed block (`1333..11333`), 10,000 replications,
  so the reported cost is out of sample (no in-sample bias)
- heuristic comparators simulated at 25,000 replications, seed 1234 (reproduce the
  published `simple`/`DP` rows within 0.17%)
- worker threads capped at 2 (`RAYON_NUM_THREADS=2`) to share the host

Result (total cost over 104 periods; `learned`/`lead_time_base_stock` CV-Normal,
`rolling_dp_s_s` Poisson published comparator):

| instance     | learned | best heuristic              | DP baseline | gap vs DP | gap vs best heur | winner       |
| ------------ | ------: | --------------------------- | ----------: | --------: | ---------------: | ------------ |
| constant_5   |  1026.3 | lead_time_base_stock 1300.4 |      1214.9 |   -15.52% |          -18.10% | learned      |
| constant_10  |  1539.0 | lead_time_base_stock 1543.2 |      1714.1 |   -10.22% |           -0.27% | learned      |
| constant_15  |  1785.2 | lead_time_base_stock 1840.1 |      2071.9 |   -13.84% |           -2.98% | learned      |
| seasonal_1   |  1517.1 | lead_time_base_stock 1546.1 |      1678.2 |    -9.60% |           -1.88% | learned      |
| seasonal_2   |  1569.9 | lead_time_base_stock 1550.0 |      1678.8 |    -6.49% |           +1.28% | lead_time_bs |
| seasonal_4   |  1534.6 | lead_time_base_stock 1559.5 |      1684.9 |    -8.92% |           -1.60% | learned      |
| growth       |  1491.0 | lead_time_base_stock 1485.8 |      1605.7 |    -7.14% |           +0.35% | lead_time_bs |
| decline      |  1711.4 | lead_time_base_stock 1654.2 |      1837.7 |    -6.87% |           +3.46% | lead_time_bs |

Findings against the table intent:

1. CMA-ES closes (and reverses) the gap to `rolling_dp_s_s`: the learned policy beats
   the DP baseline on all 8/8 instances (-6.5% to -15.5%).
2. The learned policy is the single cheapest policy on 5/8 instances (it beats the
   `simple_s_s`/`lead_time_base_stock` heuristics and DP); on seasonal_2, growth, and
   decline it still beats DP but trails `lead_time_base_stock` by 0.35%-3.46%.
3. No exact optimum exists for the rolling-forecast path; `rolling_dp_s_s` is used as
   the strongest comparator and is NOT presented as the global optimum.

Raw JSON:
`outputs/nonstationary_lot_sizing/learned_benchmark_full8_g150_p48_ac100_r10000.json`.
