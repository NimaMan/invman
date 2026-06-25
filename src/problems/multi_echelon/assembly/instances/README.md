# Assembly Instances

This directory contains Rosling-compatible equal-component-lead-time assembly instances. These are useful trainable stress cases, but they are not directly literature-verified: Rosling (1989) is a structural assembly-to-serial equivalence result, not a table of reproducible assembly costs.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `rosling_generated_4comp_lc1_poisson8.json` | `generated` | Verify by Rosling reduction to serial exact solver |
| `rosling_generated_5comp_lc2_high_penalty_poisson6.json` | `generated` | Verify by Rosling reduction to serial exact solver |
| `rosling_generated_6comp_lc4_deep_bom_poisson2.json` | `generated` | Verify by Rosling reduction to serial exact solver |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
python scripts/assembly/verify_assembly_rosling_independent.py
cargo test -q assembly
```
