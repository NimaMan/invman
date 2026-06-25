# Serial Multi-Echelon Instances

This directory is the machine-readable instance catalog for Clark-Scarf serial backorder systems. JSON files are the source of truth for instance parameters; legacy benchmark cards are presentation artifacts only.

Parameter order is explicit: literature fields use upstream-to-downstream local holding costs and lead times, while repo rollout configs usually use downstream-to-upstream installation/echelon order. The exact solver verifies all rows listed here; environment simulation is reliable only for rows whose most-downstream lead time is 1.

## Instances

| File | Classification | Verification status |
| --- | --- | --- |
| `pirhooshyaran2021_serial_case01.json` | `strict_literature` | Exact Clark-Scarf solver reproduces published row |
| `pirhooshyaran2021_serial_case05.json` | `strict_literature` | Exact-only; env simulation blocked by downstream lead time 2 |
| `pirhooshyaran2021_serial_case09.json` | `strict_literature` | Exact Clark-Scarf solver reproduces published row |

Expected validation:

```bash
python scripts/instances/validate_problem_instances.py
cargo test -q serial_rows_reproduced_by_exact_clark_scarf_solver
```
