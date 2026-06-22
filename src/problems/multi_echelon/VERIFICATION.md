# Verification Target - multi_echelon

## Primary Target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | serial Clark-Scarf / Snyder-Shen example 6.1 |
| Metric | optimal average cost |
| Literature value | `47.65` |
| Current repo value | `47.66539330768766` |
| Tolerance | `0.03` absolute |
| Last validated | `2026-06-22` |

## Source

Snyder and Shen, example 6.1 / Clark-Scarf serial multi-echelon inventory benchmark, as carried in the repo's serial subfamily benchmark. This is the cleanest strict target for the umbrella `multi_echelon` family.

Other multi-echelon subfamilies have their own caveats: divergent special-delivery carries Van Roy / Gijsbrechts context rows, general backorder reproduces some published heuristic rows, and production/assembly/distribution is partly faithful but not fully literature-number verified.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.multi_echelon_serial_exact_normal_solution(
    [3, 2, 2],
    [1, 1, 2],
    37.12,
    5.0,
    1.0,
)
print(s["optimal_cost"])
assert abs(s["optimal_cost"] - 47.65) <= 0.03
PY
```

## Notes

Because `multi_echelon` is an umbrella, this file chooses one primary strict number for future-agent smoke verification. See the subfamily `BENCHMARK.md` files when validating claims specific to divergent, assembly, PADN, or general-backorder settings.
