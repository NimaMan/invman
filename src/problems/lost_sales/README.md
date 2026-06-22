# lost_sales

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "lost_sales",
  "status": "strict_peer_reviewed_number",
  "instance": {
    "id": "vanilla_l4_p4_poisson5",
    "parameters": {
      "demand_distribution": "Poisson",
      "demand_mean": 5.0,
      "lead_time": 4,
      "holding_cost": 1.0,
      "lost_sales_penalty": 4.0,
      "fixed_order_cost": 0.0,
      "horizon": 100000,
      "seed": 123
    }
  },
  "comparator": {
    "policy": "myopic2",
    "metric": "average_cost"
  },
  "literature": {
    "value": 4.82,
    "units": "average cost",
    "source": "Zipkin (2008), Old and New Methods for Lost-Sales Inventory Systems, Operations Research 56(5):1256-1263",
    "locator": "Table 3(a), Poisson demand, penalty p=4, lead-time column L=4, p.1261",
    "url_or_doi": "https://doi.org/10.1287/opre.1070.0471"
  },
  "reproduction": {
    "current_value": 4.82075,
    "tolerance": {
      "absolute": 0.02
    },
    "last_validated": "2026-06-22",
    "command": "python -c \"import invman_rust as ir; s=ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995); got=s['myopic2']; print(got); assert abs(got - 4.82) <= 0.02\""
  }
}
```

The canonical check is the vanilla lost-sales instance from Zipkin (2008): Poisson demand with mean 5, lead time 4, holding cost 1, and lost-sales penalty 4. The published Myopic-2 average cost is 4.82 in Table 3(a), and the current Rust heuristic rollout returns 4.82075 with horizon 100000 and seed 123.

### Primary target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | `vanilla_l4_p4_poisson5` |
| Policy / metric | Myopic-2 average cost |
| Literature value | `4.82` |
| Current repo value | `4.82075` |
| Tolerance | `0.02` absolute |
| Last validated | `2026-06-22` |

### Source

Zipkin (2008), "Old and New Methods for Lost-Sales Inventory Systems", Operations Research 56(5):1256-1263, Table 3(a), Poisson demand, penalty `p=4`, lead-time column `L=4`, p.1261. DOI: `10.1287/opre.1070.0471`.

### Validation command

```bash
python -c "import invman_rust as ir; s=ir.lost_sales_heuristics_all('Poisson',5.0,0,0,0,0,4,1.0,4.0,0.0,0.0,100000,123,0.2,200,0.995); got=s['myopic2']; print(got); assert abs(got - 4.82) <= 0.02"
```

### Notes

The vanilla optimum `4.73` is a carried Zipkin DP value, not recomputed in-repo. The Myopic-2 row is the verification target because it exercises the repo's environment plus heuristic evaluator against a peer-reviewed table value. The fixed-order-cost Bijvank Table 1 exact-DP check remains a useful secondary anchor in `BENCHMARK.md`.


See `BENCHMARK.md` for the broader benchmark card and caveats.
