# Literature

Current literature anchors for `dual_sourcing`:

- Gijsbrechts et al. (2022), Section 6.2 / Figure 9
- Veeraraghavan and Scheller-Wolf (2008)
- Sheopuri et al. (2010)

Published values carried here:

- the six small-scale benchmark instances from Gijsbrechts et al. (2022)
- the Figure 9 per-instance optimality-gap labels for the main heuristic families and A3C
- no per-instance absolute optimal-cost table is publicly available in that paper

Figure 9 gap labels carried in `literature/references.rs`:

| instance | capped_dual_index | dual_index | single_index | tailored_base_surge | a3c |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dual_l2_ce105` | 0.00 | 0.11 | 0.56 | 0.06 | 0.52 |
| `dual_l2_ce110` | 0.03 | 0.18 | 1.03 | 0.99 | 0.80 |
| `dual_l3_ce105` | 0.00 | 0.27 | 0.98 | 0.01 | 0.82 |
| `dual_l3_ce110` | 0.06 | 0.36 | 2.11 | 0.71 | 0.51 |
| `dual_l4_ce105` | 0.00 | 0.36 | 1.43 | 0.00 | 1.85 |
| `dual_l4_ce110` | 0.11 | 0.49 | 2.44 | 0.58 | 1.33 |

Use `literature/references.rs` as the source of truth for:

- `PRIMARY_REFERENCE_INSTANCE`
- `DUAL_SOURCING_REFERENCE_INSTANCES`
- `FIGURE_9_GAP_REFERENCES`
- `VERIFICATION_PROBLEM_INSTANCE`
