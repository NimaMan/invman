# general_network/flownet/verification

This folder checks that the `general_network` FlowNet formulation matches the current Rust
problem semantics.

The verification surface is split into four parts:

- `structure.rs`
  - validates that the formulation answers the FlowNet questions and references a coherent event
    schedule
- `reference_alignment.rs`
  - checks that the primary diamond-network benchmark and the reduced exact-verification freeze map
    to the expected FlowNet instance parameters
- `step_semantics.rs`
  - checks that the worked-transition verification fixture and pairwise base-stock first action match the
    current `env.rs` and heuristic semantics
- `policy_performance.rs`
  - checks that the reduced exact-verification reference keeps the expected optimal and
    node-base-stock discounted costs in normal library builds, not only in test-only code paths
