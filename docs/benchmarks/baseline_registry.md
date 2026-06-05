# Baseline Registry

This page is the index for per-problem baseline registries. The registries are
small YAML files that connect published numbers, reference instances, executable
verification, repo-native gates, benchmark reports, and paper claims.

Schema: `docs/benchmarks/baseline_registry_schema.md`.

## Current Registries

| Problem | Registry | Current scope |
| --- | --- | --- |
| Vanilla lost sales | `src/problems/lost_sales/vanilla/literature/baselines.yaml` | Zipkin (2008) Table 3(a), canonical `vanilla_l4_p4_poisson5`; mixed executable heuristic rows and table-only optimal/CBS rows. |
| Lost sales with fixed order cost | `src/problems/lost_sales/fixed_order_cost/literature/baselines.yaml` | Bijvank et al. (2015) Table 1 strict verifier plus repo-native canonical `lit_pois_mu5_l4_p4_k5` gate. |
| Perishable inventory | `src/problems/perishable_inventory/literature/baselines.yaml` | De Moor/Farrington exact `m=2,L=1` slice plus one table-only `m=4,L=2` medium/practical anchor; other Scenario A rows remain in `references.rs` until explicitly promoted. |

## Rules

- Do not promote a number to `strict_literature_verified` unless a named repo
  test or command re-derives it.
- Use `partial` when an instance mixes executable rows and table-only rows.
- Use `repo_native` for benchmark values produced by this repo, even if the
  instance family is literature-inspired.
- Keep source citations and table/figure labels close to the rows they support.
- Give every promoted row a stable `row_id`, a structured source locator, and a
  BibTeX `citation_key` when `paper/references.bib` already has the source.
- Keep missing data explicit in `unknowns`; do not infer precise values from
  prose.

## First-Pass Notes

- Vanilla lost sales is `partial` in the registry because the Myopic-1,
  Myopic-2, and SVBS rows are executable, while the optimal and
  better-vector base-stock row are table-only in the current verifier. Xin
  (2021) is corroborating context for capped base-stock policy structure, not
  the load-bearing source for the Zipkin table row.
- Fixed-order-cost Table 1 is `strict_literature_verified`: the exact solver and
  exact heuristic evaluators reproduce all promoted published rows at cap 24.
- Perishable inventory has strict entries for the two exact `m=2,L=1` instances.
  Only the listed `m=4,L=2` row is promoted as a table-only anchor here; remaining
  Scenario A constants stay in `references.rs` unless a future registry update
  adds row-level source locators and verifier status.
