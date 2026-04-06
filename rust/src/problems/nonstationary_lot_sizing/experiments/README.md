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
