# Fixed-Cost Canonical Benchmark Refresh

Canonical fixed-order-cost benchmark instance:

- problem: single-item periodic-review lost sales with a fixed setup cost
- lead time `L=4`
- shortage cost `p=4`
- fixed ordering cost `K=5`
- holding cost `h=1`
- demand `~ Poisson(5)`
- reference instance: `lit_pois_mu5_l4_p4_k5`

This is the fixed-cost counterpart to the canonical vanilla lost-sales benchmark. The instance is
drawn from the Bijvank-Bhulai-Huh (2015) benchmark family, but the exact per-instance numbers
below are repo-native because the literature does not publish a clean exact-cost table for this
single instance.

## Benchmark protocol

Current canonical protocol for the paper-like benchmark suite:

- training episodes: `5000`
- CMA-ES population: `50`
- training horizon: `2000`
- evaluation horizon: `1,000,000`
- evaluation seeds: `10`
- warm-up discarded: `20%`

The suite runner is:

- [benchmark_fixed_cost_canonical_suite.py](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/scripts/benchmark_fixed_cost_canonical_suite.py)

The current suite outputs live in:

- [fixed_cost_l4_canonical_suite_5k_paperlike](/Users/nimamanaf/Desktop/code/ML/inventory_management/invman/outputs/benchmarks/fixed_cost_l4_canonical_suite_5k_paperlike)

## Heuristic baselines

Current long-run heuristic anchors on the canonical instance:

| Heuristic | Parameters | Mean cost |
| --- | --- | ---: |
| `s,S` | `s=21, S=27` | `9.37145` |
| `s,nQ` | `s=22, q=8` | `9.18096` |
| modified `s,S,q` | `s=22, S=30, q=8` | `9.17436` |

So the best heuristic anchor for this canonical instance is modified `s,S,q` at `9.17436`.

## Learned policy families

The completed fixed-cost policy matrix on the same canonical instance is:

| Backbone | Head / structure | Backend | Mean cost | Status |
| --- | --- | --- | ---: | --- |
| Linear | categorical quantity | `rust` | `10.27299` | trusted |
| Linear | gated ordinal quantity | `python` | `8.76878` | trusted |
| NN | categorical quantity | `rust` | `10.27299` | provisional |
| NN | gated ordinal quantity | `python` | `8.73282` | trusted |
| Soft tree | oblique depth-2, linear leaf | `rust` | `8.77418` | trusted |
| Soft tree | oblique depth-1, linear leaf | `rust` | `8.77846` | trusted |

Important note:

- the current `nn_categorical_quantity` run returned a value numerically identical to the linear
  categorical baseline; that row is kept for completeness but should be re-verified before relying
  on it for publication claims

## Current interpretation

The fixed-cost benchmark now supports a much sharper conclusion than the earlier exploratory runs:

- the plain categorical quantity head is not appropriate for this problem
- the main architectural improvement is in the action parameterization, not only in the backbone
- moving from `categorical_quantity` to `gated_ordinal_quantity` changes the linear policy from
  clearly non-competitive to essentially tied with the best tree variants
- the best current canonical policy is `nn_gated_ordinal_quantity` at `8.73282`
- `linear_gated_ordinal_quantity`, `soft_tree_depth2_linear_leaf`, and
  `soft_tree_depth1_linear_leaf` all cluster very tightly around `8.77`

Against the best heuristic `9.17436`, the trusted learned policies improve by roughly:

- `nn_gated_ordinal_quantity`: `4.81%`
- `linear_gated_ordinal_quantity`: `4.42%`
- `soft_tree_depth2_linear_leaf`: `4.36%`
- `soft_tree_depth1_linear_leaf`: `4.32%`

So the current fixed-order-cost evidence supports a policy-design conclusion:

- for this problem class, choosing the right action space matters more than adding nonlinear model
  capacity
