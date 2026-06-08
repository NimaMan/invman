# agentic_policy_search

Automated discovery of inventory-control **policy structures** (action heads, per-dimension
geometry, state-dependent leaves, regime splits, feature bases) by an **LLM agent — Codex,
driven through the `beden` framework** (`/home/nima/code/ai/agents/beden`) — whose proposals
are scored by the **invman Rust rollout oracle** under a seed-robust gate.

This automates the single human-bottlenecked step that `invman/autoresearch/` documents a
person doing today: *"propose one policy-structure change per round, train it, keep it only if
it beats the gate."* Here Codex proposes/recombines the structure; CMA-ES still owns the
continuous parameters; the gate/keep decision stays deterministic and seed-robust.

It is **not** a clone of BES (arXiv:2605.28814). BES recombines LLM *text trajectories* and uses
*backward goal-trees* for sparse reward — neither maps onto a continuous-parameter CMA-ES loop.
What we borrow is the **ShinkaEvolve/AlphaEvolve outer pattern** (archive → select → propose →
evaluate) implemented on `beden`, plus two *optional* secondary signals (effective-score bucket
guard, dense goal-tree) reserved for genuinely flat-cost plateau problems — off by default here.

--------------------------------------------------------------------------------------------------
## Objective (the only success metric)

Discover a policy spec that **robustly beats the in-repo echelon-base-stock gate** —
mean over **≥5 optimizer seeds**, paired CRN, held-out block — on hard instances where
hand-chosen geometries have not.

- **Primary spearhead:** OWMR (Kaynov 2024) **`instance_14`** (K=10 strongly heterogeneous,
  partial backorder). We currently only *tie* the gate (deployed = warm-start anchor; the gate
  Cartesian search is ~3×10¹⁴), yet exploitable structure is *proven*: changing the leaf/target
  head already flipped the sign on `instance_12` (+4.6% over gate, 6/6 seeds) and `instance_13`
  (+6.9%, 5/5 seeds).
- **Published PPO is cross-protocol context only.** The Kaynov PPO figures are single published
  scalars, never re-trained here; the env reproduces Kaynov's base-stock only within ~1–6% with a
  sign flip and an unverified N(μ,σ) convention on the high-CV rows. We **never** report a
  head-to-head PPO "beat". The honest, like-for-like comparator is the in-repo gate.
- **Fast-follow:** PADN mixed-network (residual gate-backbone head, identical structure blocker);
  JR high-cost settings (newsvendor order-up-to head).

--------------------------------------------------------------------------------------------------
## Why beden + Codex

- The runtime proposer **is** an agent. `beden`'s `beyin-codex` brain runs `codex exec` in a
  **ReadOnly sandbox** with a JSON output schema and returns **typed tool-requests** (normalized
  via `elayaq` into `ActionRequest`s). So Codex *only reasons and emits a structured policy-spec
  proposal*; it never touches the filesystem or runs the evaluation itself.
- All real work runs through `beden`'s **default-deny `omurga` execution gate** and is recorded in
  a typed, inspectable **transcript** (`yuz` can render the live search). This matches the lab's
  no-silent-fallback, auditable, seed-robust discipline.
- Codex's strength is code-gen, and a proposal here *is* a small policy-spec program — a natural
  fit, and tighter than vendoring any Python LLM stack.

--------------------------------------------------------------------------------------------------
## Architecture (full algorithmic description)

Outer evolution (this crate, on `beden`) + inner parameter optimization (invman CMA-ES):

```
archive A := []                                  # jsonl population of evaluated specs
seed A with the GATE spec (evaluate it once)     # gen-0 anchor = the heuristic gate
for generation g in 1..G:
    ctx  := render_context(problem_brief,
                           diverse_elites(A),    # MAP-Elites: best spec per OCCUPIED structural
                                                 #   niche (action_head|leaf_type|split_type),
                                                 #   best-first — structurally-DISTINCT parents
                           tried_signatures(A),  # every structure already tried + best cost/robust
                           best_signature(A),    # the structure to deviate from on >=1 axis
                           untried_niches(A))    # concrete unexplored DSL cells (+ coverage)
    spec := Codex.propose(ctx)        # beden run_turn -> typed tool_request (policy-spec JSON)
    res  := evaluate_policy_spec(spec)        # omurga action -> python oracle CLI (below)
    A.record(spec, res)
    log best_deployed := min over A of res.deployed_cost
return argmin spec + full beden transcript
```

**Novelty pressure (anti-plateau).** Earlier runs plateaued: with a flat `top_k`-by-cost context
the parents collapsed to ~5 near-duplicates of the single archived best, so Codex deterministically
re-emitted that exact spec (run-2 gens 1-4 re-proposed an identical depth-1 spec). The context is now
a quality-diversity + anti-repeat payload: a **structural signature** (a canonical key over
`action_head | leaf_type | split_type | depth | per_retailer_targets | features | backbone |
warm_start`, ignoring temperature/continuous params — CMA-ES owns those) is computed for every
archived spec. The prompt enforces a hard rule: **never re-propose a `tried_signature`, and differ
from `best_signature` on ≥1 structural axis, preferring an `untried_niche`**. This is context+prompt
pressure only — it preserves the one-Codex-call + one-eval per generation contract and the honest
metric (`robust_gate_beat`, `deployed_cost = min(trained, gate)`); it does not hard-reject a
duplicate at eval time. The signature/diversity helpers (`structural_signature`, `tried_signatures`,
`diverse_elites`, `untried_niches`) are pure reads over the canonical archive (no schema change).

`evaluate_policy_spec(spec)` (Python oracle, invman side):
```
1. compile  : spec(JSON, our DSL)  ->  invman.Policy        (policy_build / policy_registry)
2. warm-start: CMA-ES mean := gate-invertible anchor so generation-0 == the gate exactly
3. inner opt : CMA-ES for budget B on the Rust population-rollout oracle (rollout_fitness)
4. eval      : seed-robust held-out, >=5 seeds, paired CRN, vs the same-protocol gate
5. floor     : deployed_cost := min(trained_xbest_cost, gate_cost)   # honest deploy floor
6. return    : structured result (schema below)
```

Honest reporting is structural, not optional:
- A spec is a **robust gate-beat** iff **every** evaluation seed is below the gate AND
  `mean + std < gate_cost`. Anything weaker is logged as "parity / not robust", never a win.
- `deployed_cost` is always the better of {trained policy, gate} — we never deploy below the gate
  floor on the strength of a lucky seed.
- No silent fallbacks: a spec that fails to compile or run returns `compiled_ok=false` + the raw
  error; the loop records it and continues (the brain may use the error as evidence).

--------------------------------------------------------------------------------------------------
## Policy-spec DSL — the contract Codex emits (one tool call per proposal)

```jsonc
{
  "problem": "one_warehouse_multi_retailer",
  "instance": 14,
  "backbone": "soft_tree",          // soft_tree | linear
  "depth": 2,                        // soft_tree only
  "split_type": "oblique",          // oblique | axis_aligned
  "leaf_type": "linear",            // constant | linear   (state-dependent vs flat)
  "temperature": 0.25,
  "action_head": "echelon_targets", // echelon_targets | symmetric_echelon_targets
                                     // | echelon_targets_with_alloc_targets
                                     // | echelon_targets_with_holdback | direct_orders
  "per_retailer_targets": true,      // grow control dim to per-retailer vs one shared target
  "features": ["on_hand", "backlog", "pipeline"],   // feature basis (pipeline-aware)
  "warm_start": "gate_invertible",   // gate_invertible | none
  "rationale": "free-text reasoning from the proposer (recorded, not executed)"
}
```
Constraints the compiler enforces (feasibility — every emitted spec must be evaluable):
- `direct_orders` / raw heads are allowed but lose without a gate anchor; the compiler still
  honours `warm_start`. Decoder must clip-to-position / project onto physical caps.
- `echelon_targets` with `per_retailer_targets=true` ⇒ `control_dim = K+1` (warehouse + K retailers).
- `echelon_targets_with_holdback` (per_retailer) ⇒ `control_dim = K+2`: the K+1 echelon targets PLUS
  one SIGNED-residual warehouse-holdback control `h`. The release step rations against
  `release_capacity = max(warehouse_available − round(h).max(0), 0)`, so the held-back `h` units stay
  centrally and feed the prob-0.8 partial-backorder emergency channel (cheap central risk pooling).
  `h` decodes via the identity-leaf tail of `action_targets_with_signed_tail_from_flat_params`, so it
  is EXACTLY 0 at the gate-invertible warm-start ⇒ generation-0 reproduces the plain `echelon_targets`
  release byte-exact and the DOF can only help from there.
- Unknown enum values ⇒ `compiled_ok=false` with an explicit message (no silent coercion).

## evaluate I/O contract (oracle return)

```jsonc
{
  "compiled_ok": true,
  "mean_cost": 0.0, "std_cost": 0.0, "per_seed": [/* >=5 */], "n_seeds": 5,
  "gate_cost": 0.0, "gate_gap_pct": 0.0,            // (mean - gate)/gate * 100
  "n_seeds_below_gate": 0,
  "deployed_cost": 0.0,                             // min(trained, gate)
  "robust_gate_beat": false,                        // all seeds below gate AND mean+std < gate
  "error": null
}
```

--------------------------------------------------------------------------------------------------
## Components (file map)

- `README.md` — this algorithmic description + the two contracts (DSL, evaluate I/O).
- `policy_spec_compiler.py` — DSL(JSON) → `invman.Policy` (+ gate-invertible warm-start anchor).
- `evaluate_policy_spec.py` — the oracle CLI: compile → inner CMA-ES → ≥5-seed paired-CRN eval
  vs gate → result JSON. Reuses existing invman OWMR machinery (gate search + held-out eval).
- `agent/` — the `beden` Rust crate (the LLM agent):
  - `Cargo.toml` — path deps on beden crates (`oz`, `omurga-*`, `beyin`, `beyin-codex`, `elayaq`,
    `shared/*`). No dep on invman_rust (the oracle is reached via the Python CLI).
  - `src/main.rs` — the generation loop (`run_turn` cycles, archive, best-so-far logging).
  - `src/actions.rs` — `EvaluatePolicySpecAction` (subprocess → `evaluate_policy_spec.py`),
    `ArchiveAction` (read top-K / append to `archive.jsonl`) plus the novelty-pressure read helpers
    `structural_signature` / `tried_signatures` / `diverse_elites` / `untried_niches`.
  - `src/brain.rs` — `CodexBrainConfig` (ReadOnly sandbox), the OWMR system prompt, and the
    policy-spec JSON schema Codex must satisfy.
- `archive.jsonl` — runtime population of `{spec, result}` rows (git-ignored).
- `SMOKE.md` — one-iteration end-to-end runbook + integration punch-list.

Every source file carries a full algorithmic-description header comment (lab convention).

--------------------------------------------------------------------------------------------------
## Status

**MVP verified end-to-end (2026-06-06).** `cargo build` links the agent binary (incl. the
`llama-cpp-sys` C build); the Python oracle runs seed-robustly on OWMR `instance_14`; the full
`beden` → oracle → archive loop completes (stub brain, exit 0) with honest output — the
gate-invertible warm-start reproduces the gate at gen-0, and `robust_gate_beat` stays `false` on a
within-noise dip (1/5 seeds), so no overclaim. One integration fix was applied during verification:
the oracle keeps **stdout JSON-only** (CMA-ES/pycma logs go to stderr) so the Rust action can parse it.

Run (stub brain — exercises the real oracle, no Codex spend):

    cd invman/agentic_policy_search/agent
    APS_STUB_BRAIN=1 RAYON_NUM_THREADS=4 ./target/debug/agentic-policy-search-agent \
        --generations 1 --seeds 5 --budget tiny

Run (real Codex brain): drop `APS_STUB_BRAIN`, raise `--budget small|full` and `--generations`.

Next (the actual research run): real Codex at `--budget small/full` over more generations, letting it
propose per-retailer / state-dependent-leaf heads — the geometry the audit says flipped the sign on
instances 12/13 — and attempt a robust gate-beat on `instance_14`. PPO stays cross-protocol context.

Open polish (non-blocking, see SMOKE.md punch-list): widen the action descriptor's
number fields to `["number","null"]`; trim unused Cargo deps.
