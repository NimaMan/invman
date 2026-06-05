# Literature

Literature anchor for the vanilla (no fixed-order-cost) lost-sales family.

- Zipkin (2008), "Old and New Methods for Lost-Sales Inventory Systems," Operations Research
  56(5):1256-1263, **Table 3(a)** (Poisson, penalty `p=4`), lead-time column `4` (p.1261).
  DOI: `10.1287/opre.1070.0471`.

The Poisson demand mean is `lambda = 5` (Zipkin's experimental value, restated by
Gijsbrechts, Boute, Van Mieghem & Zhang (2022), Management Science 68(3):1885-1903, p.11).

Executable literature verification uses the canonical instance `vanilla_l4_p4_poisson5`
(`lambda=5`, `L=4`, `h=1`, `p=4`):

- a live env + heuristic rollout reproduces the published Myopic-1 (5.06), Myopic-2 (4.82) and
  Standard-vector-base-stock (5.83) average costs to within ~0.015
- assertion lives in `../heuristics/mod.rs::vanilla_heuristic_mean_costs_match_literature_numbers`
  (absolute tolerance 0.12); the policy-ordering assertion (Myopic-2 best) is alongside it
- the published optimal `4.73` is the DP value, not produced by a heuristic rollout

Policy-name to Zipkin Table 3 row mapping:

| crate policy name          | Zipkin Table 3 row            | cost (L=4, p=4) |
|----------------------------|-------------------------------|-----------------|
| `optimal`                  | Optimal                       | 4.73            |
| `myopic1`                  | Myopic                        | 5.06            |
| `myopic2`                  | Myopic-2                      | 4.82            |
| `svbs`                     | Standard vector base-stock    | 5.83            |
| `better_vector_base_stock` | Better vector base-stock      | 4.80            |

`reference_costs.rs` carries `capped_base_stock = 4.80`; this equals Zipkin's
"Better vector base-stock" row. Xin (2021), Operations Research 69(1):61-70
(DOI `10.1287/opre.2020.2019`), Table 1, reports a comparable ~4.8 capped-base-stock cost on
this instance (corroborating; the load-bearing pin is Zipkin's own row).

Use `references.rs` as the source of truth for:

- literature metadata and pinned table/page provenance
- the carried canonical validation instance (`PRIMARY_REFERENCE_INSTANCE`,
  `VERIFICATION_PROBLEM_INSTANCE`)
- published benchmark-policy names and reported numbers, with the `literature_verified` flag

Current status:

- the vanilla family is literature-verified on the published Zipkin Table 3(a) L=4 instance
  (env + Myopic-1/Myopic-2/SVBS reproduce the printed average costs)
- the wider grid in `reference_costs.rs` mixes `literature` (transcribed from Zipkin Table 3),
  `literature+computed`, and `computed` cells; only the Poisson Table-3 rows are literature
  numbers, the Geometric and MMPP2 rows are repo-computed
