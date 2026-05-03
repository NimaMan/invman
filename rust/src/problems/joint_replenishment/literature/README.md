# Literature

Current literature anchor for `joint_replenishment`:

- Vanvuchelen et al. 2020

Repo interpretation:

- the first carried slice is a small-scale multi-item setting with a shared major replenishment cost
- benchmark policies and reduced exact verification are defined against that interpretation

Use `literature/references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- the carried small-scale settings and benchmark-policy names

Status:

- the Vanvuchelen Table 2 setting definitions are public and carried here
- the paper reports relative benchmark figures, not exact per-setting rows suitable for executable
  assertions
- this package is therefore not literature-verified
