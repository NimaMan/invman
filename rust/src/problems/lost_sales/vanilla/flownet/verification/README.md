# lost_sales/flownet/verification

This folder contains verification targets for the `lost_sales` FlowNet formulation.

It is separate from `env.rs` and `rollout.rs` because the goal here is to verify that the
formulation is faithful to the existing implementation and benchmark expectations.

The verification surface is split into two parts:

- `structure.rs`
  - does the FlowNet formulation answer the required questions and reference a coherent event
    schedule?
- `policy_performance.rs`
  - executes the canonical `myopic1`, `myopic2`, and `svbs` heuristics against the Rust
    lost-sales transition semantics
  - exposes rollout-backed hooks for learned soft-tree, linear, and neural policies
  - merges heuristic and learned-policy measurements into one verification summary
  - compares their observed mean costs with the literature targets
  - keeps `optimal_reference` and `capped_base_stock` as literature-only anchors until those
    policies are implemented here

The first policy anchor set uses the repository's standard `vanilla_l4_p4_poisson5` benchmark.
