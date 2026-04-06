# Verification Targets

Executable verifier:

- `rust/src/problems/perishable_inventory/tests/verification.rs`

What the verifier asserts:

1. reference-set shape
   - 32 Scenario A instances are present
   - the primary reference instance is `de_moor2022_m2_exp2_l1_cp7_fifo`
2. figure-level exact policy recovery
   - for `de_moor2022_m2_exp1_l1_cp7_lifo`, the best base-stock level is `5`
   - for `de_moor2022_m2_exp2_l1_cp7_fifo`, the best base-stock level is `7`
   - the optimal policy tables match De Moor et al. (2022), Figure 3
3. published return recovery
   - the rounded value-iteration return is `-1553` for experiment 1
   - the rounded value-iteration return is `-1457` for experiment 2
4. observation layout stability
   - the policy-state layout matches the official observation ordering

Verification semantics:

- these are literature-backed assertions, not repo-native toy numbers
- the exact solver lives in `rust/src/problems/perishable_inventory/value_iteration_mdp.rs`
