# Practical Benchmark Spec

Current practical scope for `lost_sales_fixed_order_cost` is intentionally narrow.

The package already has:

- demand-path heuristic evaluation helpers in `heuristics.rs`
- the literature-verified small Poisson instance in `references.rs`

What is still missing for a full practical benchmark:

- a canonical medium-sized repo-native problem instance
- a stable evaluation protocol for train and holdout demand paths
- a learned-policy benchmark report under that protocol

Until that is added, this folder is the placeholder for the practical benchmark contract rather
than a completed benchmark suite.
