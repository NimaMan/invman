# Verification

The verification target for this family is a reduced finite-horizon joint pricing-and-ordering instance with:

- a three-level price ladder
- price-specific demand distributions
- periodic ordering with lost sales
- exact finite-horizon DP as the reference solver

The executable assertions live in `rust/src/problems/joint_pricing_inventory/tests/verification.rs`.
