# Verification Target - decentralized_inventory_control

## Primary Target

| Field | Value |
| --- | --- |
| Status | `reference_closed_form_number` |
| Instance | `beer_game_classic_four_stage` closed-form anchor |
| Metric | 36-week classic Beer Game total cost |
| Literature / reference value | `204.0` total, from per-stage costs `[46.0, 50.0, 54.0, 54.0]` |
| Current repo value | `204.0` total |
| Tolerance | exact, `0.0` absolute |
| Last validated | `2026-06-22` |

## Source

Sterman / Edali-Yasarcan classic anchor-and-adjust Beer Game reference as carried by the repo's closed-form port. This is a reference/closed-form reproduction, not a peer-reviewed printed cost table reproduced by the trainable `env.rs`.

## Validation Command

```bash
python - <<'PY'
import invman_rust as ir
s = ir.decentralized_inventory_control_classic_sterman_literature_summary()
print(s)
assert s["per_agent_costs"] == [46.0, 50.0, 54.0, 54.0]
assert s["total_cost"] == 204.0
PY
```

## Notes

The reusable trainable MDP does not reproduce this closed-form `204` bookkeeping value; prior audits recorded env-level costs of `378`/`278` under comparable settings. Treat `204` as a closed-form reference anchor and keep the trainable env status honest until an executable env-level literature number exists.
