# perishable_inventory/flownet/verification

This folder checks that the `perishable_inventory` FlowNet formulation matches the current Rust
problem semantics.

The verification surface is split into three parts:

- `structure.rs`
  - validates that the formulation answers the FlowNet questions and references a coherent event
    schedule
- `reference_alignment.rs`
  - checks that the primary literature reference instance maps to the expected FlowNet parameters
- `step_semantics.rs`
  - checks that FIFO and LIFO issuance change shortage, waste, and next-state evolution exactly as
    the current `env.rs` semantics imply
