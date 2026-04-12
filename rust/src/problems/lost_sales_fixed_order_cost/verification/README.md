# Verification

`lost_sales_fixed_order_cost` is verified by executable assertions in `verification/tests.rs`.

Current verifier scope:

- reference-shape checks from `references.rs`
- exact average-cost value-iteration evaluation of the published Table 1 instance
- exact evaluation of the published `(s,S)`, `(s,nQ)`, and modified `(s,S,q)` policies
- dominance checks showing the exact optimum is no worse than the published heuristic rows

The exact solver works on a bounded inventory-position state space. The current literature check
uses cap `24`, which is large enough to match the published Table 1 numbers tightly.
