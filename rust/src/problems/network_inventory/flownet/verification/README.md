# network_inventory/flownet/verification

This folder checks that the `network_inventory` FlowNet formulation matches the current Rust
problem semantics.

The verification surface is split into four parts:

- `structure.rs`
  - validates that the formulation answers the FlowNet questions and references a coherent event
    schedule
- `reference_alignment.rs`
  - checks that the primary diamond-network benchmark and the reduced exact-verification freeze map
    to the expected FlowNet instance parameters
- `step_semantics.rs`
  - checks that the worked transition reference and the node-base-stock first action match the
    current `env.rs` and heuristic semantics
- `exact_alignment.rs`
  - contains test-only exact-DP checks against the reduced verification reference; this depends on
    the current exact solver module, which is only compiled for Rust tests
