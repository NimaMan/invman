<!--
=================================================================================================
SMOKE.md -- agentic_policy_search :: one-iteration end-to-end runbook + integration punch-list
=================================================================================================
ALGORITHMIC ROLE
  This file is the operational counterpart to README.md. README states the OBJECTIVE (robustly
  beat the OWMR echelon-base-stock gate over >=5 seeds) and the two CONTRACTS (the policy-spec DSL
  Codex emits, and the evaluate I/O JSON the oracle returns). SMOKE.md tells a human/integrator how
  to (a) build the Rust agent crate, (b) run ONE generation end-to-end against OWMR instance_14,
  (c) recognise a healthy result JSON, and (d) read robust_gate_beat / deployed_cost HONESTLY.

  Two run paths are documented because the proposer brain has two modes (brain.rs):
    * STUB  brain (APS_STUB_BRAIN=1): deterministic, no Codex calls -> exercises the full loop,
      including the real Python oracle subprocess. Use this to smoke the wiring.
    * CODEX brain (default): the real beyin-codex ReadOnly proposer -> needs `codex` on PATH+authed.

  The Python oracle (evaluate_policy_spec.py) is the same in both paths; the only difference is who
  produces the spec each generation.
=================================================================================================
-->

# agentic_policy_search — SMOKE runbook (one iteration, end-to-end)

Base dir (`<base>`): `/home/nima/code/ml/invman/agentic_policy_search`
Working dir for all commands: `/home/nima/code/ml/invman` (the oracle puts this on `sys.path`).

The loop has two halves:
- **Python oracle** (`evaluate_policy_spec.py`) — compiles a DSL spec, searches+caches the gate,
  runs inner CMA-ES per seed, returns the README evaluate-I/O JSON on **stdout**.
- **Rust agent** (`agent/`) — a `beden` agent: Codex (or a stub) proposes one spec per generation,
  the agent shells out to the oracle through the omurga gate, and archives the result.

The two communicate ONLY by subprocess + stdout JSON. There is no FFI.

--------------------------------------------------------------------------------------------------
## 0. Prerequisites

- Python env with `invman` importable and the Rust binding built (the oracle imports
  `invman.cpu_limits`, `invman.es_mp`, and the in-repo OWMR script modules under
  `scripts/one_warehouse_multi_retailer/`). If `python` is not the right interpreter, every command
  below can take `--python <abs path to venv/conda python>`.
- Rust toolchain with **edition 2024** support (the agent crate and the whole `beden` graph).
- The `beden` crates present at `/home/nima/code/ai/agents/beden/crates/...` (path deps in
  `agent/Cargo.toml`). Verified present at build time.
- For the **Codex** path only: `codex` CLI on PATH and authenticated (the brain runs `codex exec`
  in a ReadOnly sandbox). The **stub** path needs none of this.

--------------------------------------------------------------------------------------------------
## 1. Sanity-check the Python oracle by itself (fastest first signal)

Run the oracle directly on the known-good MVP gate-anchor spec, tiny budget, 5 seeds. The FIRST run
searches + disk-caches the gate (instance_14 tiny ≈ 270 s once); re-runs are ≈ 7 s.

```bash
cd /home/nima/code/ml/invman
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python agentic_policy_search/evaluate_policy_spec.py \
  --spec agentic_policy_search/specs/gate_anchor_echelon_targets.json \
  --problem one_warehouse_multi_retailer --instance 14 --seeds 5 --budget tiny --workers 4
```

**stdout** is the contract JSON (one object). **stderr** carries CMA-ES progress logs. A healthy
result for the gate anchor at tiny budget looks like `outputs/smoke_i14_tiny.json`:

```jsonc
{
  "compiled_ok": true,
  "mean_cost": 51156.23, "std_cost": 0.0,
  "per_seed": [51156.23, 51156.23, 51156.23, 51156.23, 51156.23],   // len == n_seeds == 5
  "n_seeds": 5,
  "gate_cost": 51156.23, "gate_gap_pct": 0.0,
  "n_seeds_below_gate": 0,
  "deployed_cost": 51156.23,        // == min(mean trained, gate) -> the gate floor here
  "robust_gate_beat": false,        // honest TIE: nothing strictly below the gate
  "error": null
  // + provenance keys: instance, policy_action_mode, warm_started, anchor_cost, ...
}
```

Healthy-result checklist (the contract invariants):
- `compiled_ok == true`, `error == null`.
- `n_seeds == 5` and `len(per_seed) == 5` (the ≥5-seed mandate; the CLI rejects `--seeds < 5`).
- `warm_started == true` and `anchor_cost == gate_cost` exactly for a `gate_invertible` target head
  (generation-0 reproduces the gate bit-for-bit).
- `deployed_cost == min(mean_trained_cost, gate_cost)` — the honest deploy floor.

Error-path spot checks (all return **exit 0**, `compiled_ok=false`, raw `error`):
```bash
python agentic_policy_search/evaluate_policy_spec.py --spec agentic_policy_search/specs/bad_unknown_action_head.json     --instance 14 --seeds 5 --budget tiny   # unknown enum
python agentic_policy_search/evaluate_policy_spec.py --spec agentic_policy_search/specs/bad_symmetric_on_asymmetric.json --instance 14 --seeds 5 --budget tiny   # infeasible head on i14
```
Harness-precondition checks (**exit 2**, still a JSON on stdout):
```bash
python agentic_policy_search/evaluate_policy_spec.py --spec agentic_policy_search/specs/gate_anchor_echelon_targets.json --problem dual_sourcing --seeds 5 --budget tiny  # wrong problem
python agentic_policy_search/evaluate_policy_spec.py --spec agentic_policy_search/specs/gate_anchor_echelon_targets.json --instance 14 --seeds 3 --budget tiny             # <5 seeds
```

--------------------------------------------------------------------------------------------------
## 2. Build the Rust agent crate

```bash
cd /home/nima/code/ml/invman/agentic_policy_search/agent
cargo check    # type/borrow validation across the whole beden graph (reported clean)
cargo build    # final link of the bin `agentic-policy-search-agent` (NOT yet run end-to-end)
```

> The first `cargo build` is heavy: `beden-plug` transitively pulls `beyin-provider -> beyin-llama`
> (a `llama-cpp-sys` C build via cmake/cc). This is expected. If you only need the loop wiring and
> not the LLM provider weight, see the punch-list note about narrowing the plug deps.

--------------------------------------------------------------------------------------------------
## 3. Run ONE generation end-to-end — STUB brain (recommended first run)

The stub proposer emits a deterministic OWMR spec (no Codex), so this exercises the entire loop
*and* the real Python oracle subprocess. Gen-0 always evaluates the gate-anchor spec directly.

```bash
cd /home/nima/code/ml/invman/agentic_policy_search/agent
APS_STUB_BRAIN=1 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  cargo run -- --generations 1 --seeds 5 --budget tiny --python python
```

What healthy looks like on **stderr** (logs):
```
[brain] APS_STUB_BRAIN is set -> using deterministic StubProposerBrain (NO Codex calls)
[agentic-policy-search-agent] problem=one_warehouse_multi_retailer instance=14 seeds=5 budget=tiny generations=1
[gen 0] seeding archive with the GATE anchor spec...
[gen 0] gate anchor deployed_cost=51156.2344 gate_cost=Some(51156.2344)
[gen 1] proposing (archive has 1 ranked rows in context)...
[gen 1] turn stop_reason=... steps=2
[gen 1] deployed_cost=51156.2344 robust_gate_beat=false best_so_far=51156.2344
```
and on **stdout** the summary block (`================ agentic_policy_search :: summary ====...`)
with `best deployed_cost`, `robust gate-beats in archive: 0/N`, and the best spec pretty-printed.

Side effect: `agent/archive.jsonl` gains one `{spec, result, generation, ts_ms}` row per generation
(plus gen-0). Tail it to inspect: `tail -n +1 agent/archive.jsonl`. Delete it to reset the run.

> Use `--budget tiny` for the wiring smoke (fast). Use `--budget small` for the real screening MVP
> (the documented default), `--budget full` for production-grade evaluation.

--------------------------------------------------------------------------------------------------
## 4. Run ONE generation end-to-end — real CODEX brain

Requires `codex` on PATH and authenticated. The brain runs ReadOnly in `<base>` so it can read this
README/DSL, and emits exactly one `invman.evaluate_policy_spec` tool call per generation.

```bash
cd /home/nima/code/ml/invman/agentic_policy_search/agent
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  cargo run -- --generations 1 --seeds 5 --budget small        # add --model <name> to override
```
The stderr/stdout shape is identical to the stub path; only the gen-1 spec is Codex-authored. The
keep/gate decision is still deterministic and lives entirely in the oracle.

--------------------------------------------------------------------------------------------------
## 5. Reading the result HONESTLY (lab mandate — do not skip)

- **`robust_gate_beat`** is the ONLY "win" signal. It is `true` iff **every** seed is strictly below
  the gate **AND** `mean_cost + std_cost < gate_cost`. Anything weaker — a better mean, a few seeds
  under, a lucky single seed — is **parity / not robust**, reported as `robust_gate_beat=false`.
  Never quote a mean-only or best-of-N improvement as a beat.
- **`deployed_cost`** is `min(mean trained-xbest cost, gate_cost)` — the honest deploy floor. A spec
  can never "deploy" below the gate on the strength of a lucky seed. On instance_14 at tiny budget,
  `deployed_cost == gate_cost` (an honest tie), exactly as `outputs/smoke_i14_tiny.json` shows.
- **Report mean ± std over ≥5 seeds**, never a single or best-of-N number (`per_seed` is provided so
  you can recompute). The CLI enforces `--seeds >= 5` (exit 2 otherwise).
- **PPO is NOT in the schema and is NOT a comparator.** The only honest comparator is the in-repo
  echelon-base-stock gate.

--------------------------------------------------------------------------------------------------
## 6. Contract reconciliation (verified)

- Rust `EvaluatePolicySpecAction::execute` invokes:
  `<python> <oracle> --spec <tmp> --problem <p> --instance <n> --seeds <s> --budget <b>`
  — these are EXACTLY the flags `evaluate_policy_spec.py`'s argparse accepts (the optional
  `--sigma_init` / `--workers` / `--output_json` default sensibly when omitted).
- Rust reads `deployed_cost, robust_gate_beat, gate_cost, compiled_ok, error` — all present in the
  oracle's emit and in the README evaluate-I/O contract. All 11 contract keys are emitted in both
  the success and the `compiled_ok=false` result.
- Exit-code contract: 0 = ran (incl. `compiled_ok=false`); 2 = harness precondition (wrong
  `--problem`, `--seeds<5`); 1 = harness failure. The Rust action treats any non-zero exit as an
  infrastructure failure and surfaces stdout+stderr raw. The Rust binary validates `--seeds>=5` and
  defaults `--problem` to `one_warehouse_multi_retailer`, so exit 2 cannot arise in normal use.

See the punch-list at the end of this directory's review for the remaining build/run steps.
