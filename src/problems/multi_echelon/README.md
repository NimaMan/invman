# Multi-Echelon

## Verification target

The fenced block is the machine-readable contract. The sections below it are the human-readable audit trail: what instance is built, which literature/reference number is used, and how the repo-generated number is checked.

```json verification-target
{
  "schema_version": 1,
  "problem": "multi_echelon",
  "instance_id": "serial_clark_scarf_snyder_shen_example_6_1",
  "instance_parameters": {
    "network": "serial"
  },
  "policy": "serial_dp",
  "metric": "optimal_average_cost",
  "expected_value": 47.65,
  "reference": {
    "citation": "Snyder and Shen serial Clark-Scarf example 6.1",
    "locator": "Example 6.1 optimal serial-system cost",
    "doi_or_url": null,
    "literature_verified": true,
    "notes": "Primary strict target for the multi_echelon umbrella family."
  },
  "code_value": 47.66539330768766,
  "tolerance": {
    "absolute": 0.03
  },
  "command": "python - <<'PY'\nimport invman_rust as ir\ns = ir.multi_echelon_serial_exact_normal_solution(\n    [3, 2, 2],\n    [1, 1, 2],\n    37.12,\n    5.0,\n    1.0,\n)\nprint(s[\"optimal_cost\"])\nassert abs(s[\"optimal_cost\"] - 47.65) <= 0.03\nPY"
}
```

### Primary target

| Field | Value |
| --- | --- |
| Status | `strict_peer_reviewed_number` |
| Instance | serial Clark-Scarf / Snyder-Shen example 6.1 |
| Metric | optimal average cost |
| Literature value | `47.65` |
| Current repo value | `47.66539330768766` |
| Tolerance | `0.03` absolute |
| Last validated | `2026-06-22` |

### Source

Snyder and Shen, example 6.1 / Clark-Scarf serial multi-echelon inventory benchmark, as carried in the repo's serial subfamily benchmark. This is the cleanest strict target for the umbrella `multi_echelon` family.

Other multi-echelon subfamilies have their own caveats: divergent special-delivery carries Van Roy / Gijsbrechts context rows, general backorder reproduces some published heuristic rows, and production/assembly/distribution is partly faithful but not fully literature-number verified.

### Validation command

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

### Notes

Because `multi_echelon` is an umbrella, this file chooses one primary strict number for future-agent smoke verification. See the subfamily `BENCHMARK.md` files when validating claims specific to divergent, assembly, PADN, or general-backorder settings.

This folder contains multiple multi-echelon problem formulations that should stay separate because
they do not share the same dynamics or benchmark contract.

## Subproblems

Each subproblem is a distinct *version* of the multi-echelon problem (different topology and/or
contract). More versions can be added here as siblings.

- `serial/`
  - the textbook serial system (Clark & Scarf 1960): N stages in series, echelon base-stock optimal
- `assembly/`
  - the textbook assembly system (Rosling 1989): components assembled into a finished product;
    equivalent to a serial system
- `production_assembly_distribution_network/`
  - the Pirhooshyaran & Snyder (2021) general acyclic supply network: raw + finished inventory,
    `single`/`assembly`/`distribution` nodes, pairwise order-up-to decisions (the most general
    topology here; was previously the top-level `network_inventory` family)
- `divergent_special_delivery/`
  - Van Roy / Gijs one-warehouse-multi-retailer family with same-day special delivery
- `general_backorder_fixed_cost/`
  - Geevers/CardBoard Company general-network family with backorders and unit lead times

## Verification Status

- `serial/` — **literature-verified**. The `env.rs` simulation under the optimal echelon
  base-stock policy reproduces the published optima (Snyder & Shen Example 6.1 cost 47.65; discrete
  Poisson 3-stage 72.04, 2-stage 16.80, 1-stage 4.22) within Monte-Carlo error, and the `exact`
  solver reproduces them analytically (within 0.05%, cross-checked against `stockpyl.ssm_serial`).
  (Verified for demand-facing lead time 1; downstream lead time >= 2 is a known open env item.)
- `assembly/` — **NOT literature-verified — verified BY EQUIVALENCE only**
  (`literature_verified = false` on every carried instance in `assembly/references.rs`). Rosling
  (1989) proves an assembly system is equivalent to a serial system, and the env-sim under the
  optimal echelon base-stock policy reproduces the exact serial optimum from the verified serial
  solver (finished/demand-facing lead time 1; component/upstream lead times >= 2 supported). But
  there is NO directly reproducible PUBLISHED assembly number: Rosling (1989) is a structural result
  (no worked cost/base-stock table), and the only published number in the chain (Snyder & Shen
  Example 6.1 = 47.65) is a 3-stage serial system the 2-stage assembly reduction cannot reach. So
  the honest status is: structural/equivalence literature-verification (Rosling) + env reproduction
  of the (literature-verified) serial solver's optimum — the assembly instance numbers themselves
  (22.759 / 52.536 / 27.530) are solver-derived, not published. Guarded by
  `assembly::references::tests::no_assembly_instance_is_literature_verified`.
- `production_assembly_distribution_network/` — not literature-verified
  - implements the richer Pirhooshyaran model (per-node production step + pipeline holding), which
    does not reduce to the textbook serial/assembly optima; the paper's general-network simulation
    protocol could not be recovered from public sources. Single-node newsvendor rows are reproduced
    analytically. See its README and `serial_echelon_simulation.rs` for the structural gap.
- `divergent_special_delivery/` — not literature-verified
  - literature benchmark rows are carried from Van Roy and Gijs
  - the current repo implementation does not reproduce those rows tightly enough to claim
    literature verification
- `general_backorder_fixed_cost/` — not literature-verified
  - literature benchmark rows are carried from Geevers
  - set 1 is close, but the current repo implementation does not reproduce sets 2 and 3, so this
    formulation is also not literature-verified

## Structure Rule

The root `multi_echelon/` folder should only contain:

- formulation subfolders
- the root `mod.rs`
- the root `bindings.rs`
- this overview README

Any formulation-specific literature, verification, reports, or experiments belong inside the relevant
subproblem folder, not at the root.
