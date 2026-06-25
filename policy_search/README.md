# policy_search — automated & assisted policy-structure search

This folder is the **policy-structure search** subsystem for the invman benchmark. It searches the
space of *policy architectures* (action geometry, tree splits/leaves, regime gates, warm-starts) for
each inventory problem, scored by the invman Rust rollout oracle under a **seed-robust keep/discard
gate** (the strongest in-repo heuristic). It does **not** search the action *design* itself — that is
the human-authored per-problem coordinate system documented in
[`POLICY_DESIGN_GUIDELINES/README.md`](POLICY_DESIGN_GUIDELINES/README.md); this subsystem searches *structures*
expressed in that coordinate system.

It is the merge of two formerly-separate folders that were the **same topic at two automation levels**:

- the **manual** loop (`autoresearch/`): a person reads a per-problem `program_*.md` contract, edits
  the search surface, runs `scripts/<problem>/autoresearch_*.py`, logs a TSV row, updates priors.
- the **agentic** loop (`agentic_policy_search/`, codename **Evrim** — Turkish "evolution"): a
  beden+Codex LLM agent proposes/recombines the *same* structure space, scored by the *same* oracle
  and *same* gate. Its own framing: it *"automates the single human-bottlenecked step that
  autoresearch documents a person doing today."*

> **Designing a policy for a new problem? Read [`POLICY_DESIGN_GUIDELINES/README.md`](POLICY_DESIGN_GUIDELINES/README.md) first** —
> the reusable recipe: anchor the env to published costs → treat the action parameterization as part
> of the policy → encode in the best heuristic's coordinate system → warm-start CMA-ES → run the
> search loop → checklist for adding a new problem.

## Layout

```
policy_search/
  README.md                     # this index
  POLICY_DESIGN_GUIDELINES/README.md   # the shared, canonical policy-design recipe (read first)
  programs/                     # 12 per-problem "program" contracts (the manual loop's instructions)
  agentic/                      # Evrim: the LLM agent that automates the loop
  studies/                      # finished one-off search/ablation studies (read-only provenance)
```

### `programs/` — per-problem research contracts (the manual loop)
One `program_<problem>.md` per family. Each fixes: the trusted benchmark instance(s), the
strongest-heuristic keep/discard gate, the published anchor, the editable search surface (tree
depth/temperature/split/leaf + action design + CMA-ES warm-start), the Rust binding name, and an
"outcome / what we know" priors section. The **runners** these drive live in
`scripts/<problem>/autoresearch_*.py` (fixed-budget CMA-ES experiment loops that log TSV ledgers to
`outputs/autoresearch/`). Programs present: ameliorating_inventory, dual_sourcing, fixed_order_cost,
general_backorder_fixed_cost, joint_replenishment, lost_sales, multi_echelon, multi_echelon_serial,
one_warehouse_multi_retailer, perishable_inventory, production_assembly_distribution_network,
vendor_managed_inventory.

### `agentic/` — Evrim, the LLM automation
An AlphaEvolve/ShinkaEvolve-style outer evolution loop: one Codex proposal + one evaluation per
generation, archived as a JSONL row, with MAP-Elites novelty pressure rebuilt from the archive.
End-to-end:

```
LLM (Codex) proposes a policy-spec (DSL JSON)              [agent/ : Rust beden+Codex agent]
  -> compiled to an invman.Policy + gate-invertible warm-start   [policy_spec_compiler*.py]
  -> inner CMA-ES on the Rust rollout oracle, then >=5-seed paired-CRN held-out eval vs the gate
                                                                  [evaluate_policy_spec*.py]
  -> {deployed_cost, robust_gate_beat, ...} appended to archive*.jsonl
  -> novelty-pressure context (diverse elites + anti-repeat) fed to the next proposal
```

- The Rust agent (`agentic/agent/`) is **generic and multi-problem** (one binary, problem-routed
  prompts/niches/archives); it depends on the external `beden` framework and shells out to the
  `codex` CLI. It locates its Python oracle / gate-spec / archive from a single `BASE_DIR` constant
  in `agent/src/main.rs` (now `…/policy_search/agentic`).
- The Python oracles (`evaluate_policy_spec*.py`, `policy_spec_compiler*.py`) are **per-problem**
  (the `_padn` pair is a fork of the OWMR pair). They reach the Rust rollout via the invman Python
  builders and the per-problem `scripts/<problem>/` modules.
- Honest metric: `robust_gate_beat` (every seed below gate AND mean+std < gate); `deployed_cost =
  min(trained, gate)`. PPO is cross-protocol context, never a head-to-head beat.
- See `agentic/README.md`, `agentic/SMOKE/README.md`, and the two result logs
  `agentic/RESULTS_instance14/README.md` (OWMR instance_14 −12.57%, 10/10 seeds) and
  `agentic/RESULTS_padn_mixed/README.md` (PADN mixed −2.20%, 5/5 seeds).

### `studies/` — finished one-off studies (provenance)
Conclusions already folded into the `programs/` files and project memory; kept as read-only
write-ups, not an active surface:
- `studies/dual_sourcing_policy_search/` — the six-row Gijs Fig-9 factor screen (control geometry >
  parameter count; capped-dual-index + small regular-order caps).
- `studies/fixed_cost_ordinal_stability/` — root-cause note on the ordinal-head state-scaling drift.
- `studies/replenishment_geometry_search/` — the lost-sales linear-head geometry study.

## Method (shared by both loops)

- **One trusted benchmark instance** per problem, **one narrow editable surface**, **one fixed
  simulation budget** (screening to reject weak ideas, promoted full-budget for promising ones),
  **automatic logging**, **keep/discard vs a running baseline**. Adapted from
  `karpathy/autoresearch`, but budget is fixed in *rollouts* (not wall-clock) so policy classes with
  different backends compare fairly.
- **Gate = the strongest in-repo heuristic**, evaluated on a held-out common-random-number block.
- **Seed-robust mandate**: report mean±std over ≥5 optimizer seeds; a single-seed or best-of-N
  "beat" is `at_risk`, not a robust result.

## Headline results (snapshot — authoritative numbers live in the `programs/` outcome sections)

| Problem | Result | Gate |
|---|---|---|
| one_warehouse_multi_retailer (instance_14) | −12.57% robust (10/10 seeds, Evrim) | tuned echelon base-stock + allocation |
| production_assembly_distribution_network (mixed) | −2.20% robust (5/5 seeds, Evrim residual head) | env's own best pairwise base-stock |
| general_backorder_fixed_cost (Geevers set-1) | −24.3% robust (5/5 seeds, paper-table TSV rows) | published node base-stock ~10,355 |
| multi_echelon (Gijs settings 1&2, direct_level) | ~−14.4% (> published A3C gap) | best in-env constant base-stock |
| lost_sales fixed-order-cost (canonical) | 8.776 (50k) vs heuristic 9.165 | s,S / s,nQ / modified s,S,q |
| lost_sales vanilla | 4.754 (oblique depth-2 linear-leaf) | Zipkin myopic family |

These are search outcomes, not literature verification. See each `programs/program_<problem>.md` for
the per-row provenance, seed count, and `at_risk` status.
