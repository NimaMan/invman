# Verification Target - lost_sales

## Primary Target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | `bijvank2015_table1_l2_p14_k5` |
| Metric | average cost, exact fixed-order-cost lost-sales DP |
| Literature value | optimal `11.46`; `(s,S)` `11.62`; `(s,nQ)` `11.56`; modified `(s,S,q)` `11.50` |
| Current repo value | optimal `11.463052002030395`; `(s,S)` `11.61814785131375`; `(s,nQ)` `11.555215531299211`; modified `(s,S,q)` `11.497402734121692` |
| Tolerance | `0.005` absolute for all four rows |
| Last validated | `2026-06-22` |

## Source

Bijvank, Bhulai, and Huh (2015), "Parametric replenishment policies for inventory systems with lost sales and fixed order cost", European Journal of Operational Research 241(2):381-390, Table 1.

Secondary vanilla lost-sales anchor: Zipkin (2008), "Old and New Methods for Lost-Sales Inventory Systems", Operations Research 56(5):1256-1263, Table 3(a), carries Poisson `L=4`, `p=4`, `h=1`, `mu=5` costs including optimal `4.73`, myopic `5.06`, myopic-2 `4.82`, SVBS `5.83`, and better vector base-stock `4.80`. The repo re-runs the heuristic rows; the optimal `4.73` is carried as a published DP value.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.lost_sales_fixed_order_cost_exact_literature_summary(
    "bijvank2015_table1_l2_p14_k5",
    24,
)
checks = [
    ("optimal_average_cost", "published_optimal_cost"),
    ("s_s_average_cost", "published_s_s_cost"),
    ("s_nq_average_cost", "published_s_nq_cost"),
    ("modified_s_s_q_average_cost", "published_modified_s_s_q_cost"),
]
for got_key, pub_key in checks:
    got, pub = s[got_key], s[pub_key]
    print(got_key, got, "published", pub, "gap", got - pub)
    assert abs(got - pub) <= 0.005
PY
```

## Notes

Use the fixed-order-cost Table 1 row as the canonical exact validation because it compares an in-repo exact DP value to a peer-reviewed table value. Use the Zipkin row as a secondary vanilla heuristic validation.
