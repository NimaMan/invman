#!/usr/bin/env python
# =============================================================================
# evaluate_policy_spec_padn.py -- the Python ORACLE CLI for the PADN fast-follow
# =============================================================================
# OBJECTIVE
#   Score ONE policy-spec proposal (the PADN DSL JSON the brain emits) on the MIXED
#   production_assembly_distribution_network instance (Pirhooshyaran & Snyder 2021,
#   Fig. 1 / Table 5) under the lab's honest, seed-robust gate-beat metric, and print
#   the README evaluate-I/O JSON to stdout. This is the PADN sibling of
#   evaluate_policy_spec.py (the OWMR oracle): it is invoked as a subprocess by the
#   agent (omurga action) once per generation and its stdout JSON is recorded in the
#   archive.
#
#   PADN HAS NO PUBLISHED DRL/PPO BASELINE. The ONLY comparator is the in-repo
#   env-own pairwise base-stock GATE (grid-searched per-echelon order-up-to levels,
#   re-scored on a disjoint held-out CRN block). PPO is NOT in the schema and is NOT
#   compared -- there is no cross-protocol context here, just the gate.
#
#   CLI (mirrors the OWMR oracle's flag contract; the agent ALWAYS passes
#        --spec --problem --instance --seeds --budget, so --instance is parsed and
#        IGNORED -- PADN is a single fixed instance and the agent's fixed OWMR-shaped
#        call must work unchanged):
#     python evaluate_policy_spec_padn.py --spec spec.json \
#         --problem production_assembly_distribution_network --instance 0 \
#         --seeds 5 --budget smoke
#
# FULL ALGORITHMIC DESCRIPTION (matches the OWMR oracle's evaluate_spec, PADN-fitted)
#   0. parse args; hard CPU cap from argv BEFORE numpy / Rust import (lab convention).
#      --instance is parsed and IGNORED: the PADN problem is the single mixed SCN
#      defined in the base module (autoresearch_mixed_distribution_assembly_network);
#      there is no instance dimension, so whatever the agent passes is discarded (never
#      a failure -- the agent's fixed --instance call must succeed).
#   1. COMPILE  : spec(JSON) -> CompiledPadnPolicy via policy_spec_compiler_padn (the
#                 separate, validating DSL compiler -- mirrors evaluate_policy_spec.py
#                 importing policy_spec_compiler). A bad spec (unknown enum / wrong
#                 problem / bad depth / learning the backbone) returns
#                 {compiled_ok:false, error:<raw>} with EXIT 0 (the loop uses the error
#                 as evidence). Only a genuine harness failure exits non-zero.
#   2. GATE     : grid-search the env-own pairwise base-stock gate ONCE on the SEARCH
#                 CRN block and re-score the argmin on the HELD-OUT block, reusing the
#                 base module's search_best_pairwise_base_stock VERBATIM (the SAME gate
#                 the production seed-robust runner reports). gate_cost = held-out mean.
#                 gate_oul = the integer per-relation order-up-to vector (= the residual
#                 backbone_levels, length ACTION_DIM = 8).
#   3. ANCHOR (gen-0 == gate guarantee, end-to-end). For the residual gate-backbone head
#                 order = clamp(gate_order + round(Delta)) and Delta(zeros) == 0, so the
#                 residual-zero warm start reproduces the gate BYTE-EXACT. The oracle
#                 computes, on the held-out block:
#                   anchor_cost = soft_tree_cost_on_paths(zeros, ...,
#                                   action_mode='residual_base_stock',
#                                   backbone_levels=gate_oul, residual_group_of=...)
#                 and ASSERTS |anchor_cost - gate_cost| < 1e-6. If that fails the oracle
#                 EXITS NON-ZERO with HARNESS_FAILURE (no silent fallback) -- a broken
#                 gate-invertibility invariant is an infrastructure bug, not a spec error.
#                 For action_head=='vector_quantity' there is no exact gate anchor
#                 (raw-order head); anchor_cost is None and the honest floor (step 4) keeps
#                 the spec downside-safe at the gate.
#   4. INNER OPT: for EACH of the >=5 optimizer seeds, run CMA-ES (smoke/screening/full
#                 budget) on the Rust population-rollout binding with the warm-started x0,
#                 threading the compiled action_mode + backbone_levels(gate_oul) +
#                 residual_group_of through base.population_costs. Evaluate the trained
#                 xbest on the SAME held-out block (paired CRN). The per-seed deployed cost
#                 is the honest floor = min over real-rollout costs {trained xbest, anchor,
#                 gate} -- we never let a degenerate CMA seed report worse than the gate.
#   5. ROBUST METRIC (the README contract, computed structurally, keys + formula VERBATIM
#                 from the OWMR oracle):
#                 per_seed              = [deployed-per-seed cost] (len == n_seeds)
#                 mean_cost / std_cost  = mean / population std of per_seed
#                 gate_cost             = held-out best-echelon gate cost
#                 gate_gap_pct          = (mean_cost - gate_cost)/gate_cost*100
#                 n_seeds_below_gate    = #{s : per_seed[s] < gate_cost}
#                 robust_gate_beat      = (n_seeds_below_gate == n_seeds) AND
#                                         (mean_cost + std_cost < gate_cost)
#                 deployed_cost         = min(mean trained-xbest cost, gate_cost)
#   6. PRINT the result JSON to stdout (one object, the README schema) and exit 0.
#
# HONEST REPORTING (lab mandate, structural here -- mirrors the OWMR oracle)
#   - >=5 seeds always (the CLI rejects --seeds < 5 as a harness error, exit != 0).
#   - paired CRN: gate, anchor and every learned seed are scored on the identical
#     held-out demand-path block (HOLDOUT_SEED via the base module).
#   - robust_gate_beat is the conjunction above; anything weaker is robust_gate_beat=false.
#   - deployed_cost is the gate floor; a spec can never deploy below the gate.
#   - NO PPO / DRL baseline. The gate is the only comparator (gate-beat, not PPO).
#
# STDOUT IS THE JSON-ONLY CONTRACT CHANNEL (same discipline as the OWMR oracle): the
#   agent action parses the whole of stdout as the evaluate-I/O JSON. The inner CMA-ES
#   logs via print()/sys.stdout, so we redirect Python-level stdout to stderr for the
#   entire computation and emit the result JSON to the RESTORED real stdout only.
#
# BUDGETS reuse the base module's BUDGETS (smoke/screening/full) verbatim so the held-out
#   block and gate grid are byte-identical to the production runner; we add the CMA-ES
#   generation counts per tier (the base BUDGETS already carry popsize/train_batch/
#   search_paths/holdout_paths/grid).
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

# --- hard CPU cap BEFORE numpy / Rust import (lab convention) --------------- #
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PADN_SCRIPT_DIR = _PACKAGE_ROOT / "scripts" / "production_assembly_distribution_network"
_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_PACKAGE_ROOT), str(_PADN_SCRIPT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from invman.cpu_limits import (  # noqa: E402
    bounded_worker_count,
    configure_process_cpu_limits_from_argv,
)

configure_process_cpu_limits_from_argv(sys.argv[1:], default=4)

import numpy as np  # noqa: E402

# The mixed-SCN instance constants, gate search, warm-start, and rollout helpers all
# live in the base module so the topology / parameters / CRN blocks / budgets / gate are
# BYTE-IDENTICAL to the production runner. We REUSE them; we do not re-derive geometry.
import autoresearch_mixed_distribution_assembly_network as base  # noqa: E402
from invman.cmaes import CMAES  # noqa: E402
from policy_spec_compiler_padn import (  # noqa: E402
    CompiledPadnPolicy,
    PolicySpecError,
    compile_padn_spec,
)

PROBLEM = "production_assembly_distribution_network"
ANCHOR_TOL = 1e-6  # |anchor_cost - gate_cost| must be below this (gen-0 == gate guarantee)

# Inner CMA-ES generation counts per budget tier (the base BUDGETS already carry popsize /
# train_batch / search_paths / holdout_paths / grid; we only add the generation budget so
# the smoke tier is fast and the full tier mirrors the seed-robust runner).
CMA_GENERATIONS = {"smoke": 8, "screening": 40, "full": 60}
DEFAULT_SIGMA = {"smoke": 0.2, "screening": 0.2, "full": 0.2}

# Per-seed optimizer seeds derive from this base (matched in spirit to the OWMR oracle's
# fixed seed schedule so runs are reproducible given --seeds).
SEED_BASE = 700_000
SEED_STEP = 101


def _empty_result(error: str) -> dict:
    """README evaluate-I/O object for a spec that failed to compile/run. Keys VERBATIM
    match the OWMR oracle (and the README contract)."""
    return {
        "compiled_ok": False,
        "mean_cost": None,
        "std_cost": None,
        "per_seed": [],
        "n_seeds": 0,
        "gate_cost": None,
        "gate_gap_pct": None,
        "n_seeds_below_gate": 0,
        "deployed_cost": None,
        "robust_gate_beat": False,
        "error": str(error),
    }


def _train_one_seed(
    compiled: CompiledPadnPolicy, budget, generations, gate_oul, x0, seed, sigma_init
) -> dict:
    """Train ONE CMA-ES seed on the population-rollout binding; return its trained xbest
    + best_train. Mirrors the seed-robust runner's _train_one_seed but threads the
    compiled action_mode + backbone_levels(gate_oul) + residual_group_of through the
    base helpers so the inner optimization uses the residual head."""
    depth = compiled.depth
    leaf = compiled.leaf_type
    split = compiled.split_type
    temp = compiled.temperature
    action_mode = compiled.policy_action_mode
    group_of = compiled.residual_group_of

    n = int(base._flat_param_count(depth, leaf))
    es = CMAES(num_params=n, sigma_init=float(sigma_init), popsize=int(budget["popsize"]),
               seed=int(seed), x0=list(x0))
    rng = np.random.default_rng(int(seed) + 1)
    train_batch = int(budget["train_batch"])
    best_flat = np.asarray(x0, dtype=np.float64).copy()
    best_train = math.inf
    t0 = time.time()
    for _ in range(int(generations)):
        sols = es.ask()
        b = int(rng.integers(1, 10_000_000))
        seeds = list(range(b, b + train_batch))  # paired CRN within the generation
        rewards = []
        for k in range(es.popsize):
            batch = [sols[k].astype(np.float32).tolist()] * train_batch
            cost = float(
                base.population_costs(
                    batch, depth, leaf, split, temp, seeds,
                    action_mode=action_mode,
                    backbone_levels=gate_oul,
                    residual_group_of=group_of,
                ).mean()
            )
            rewards.append(-cost)
        es.tell(rewards)
        gi = int(np.argmax(rewards))
        if -rewards[gi] < best_train:
            best_train = -rewards[gi]
            best_flat = sols[gi].copy()
    train_seconds = time.time() - t0
    return {"best_flat": best_flat, "best_train": float(best_train),
            "train_seconds": float(train_seconds)}


def evaluate_spec(spec: dict, n_seeds: int, budget_name: str,
                  sigma_init: float | None) -> dict:
    """Compile, search the gate, anchor-check (residual gen-0==gate), train (>=n_seeds),
    evaluate seed-robustly vs the gate, and return the README evaluate-I/O dict. Raises
    only on a genuine harness failure (caught by main -> non-zero exit); a non-compiling
    spec returns _empty_result with compiled_ok=false (exit 0)."""
    budget = base.BUDGETS[budget_name]
    generations = CMA_GENERATIONS[budget_name]
    sigma = DEFAULT_SIGMA[budget_name] if sigma_init is None else float(sigma_init)

    # ---- 1. COMPILE (spec errors => compiled_ok=false, NOT a harness failure) ---
    try:
        compiled = compile_padn_spec(spec)
    except PolicySpecError as exc:
        return _empty_result(f"PolicySpecError: {exc}")

    depth = compiled.depth
    leaf = compiled.leaf_type
    split = compiled.split_type
    temp = compiled.temperature
    action_mode = compiled.policy_action_mode
    group_of = compiled.residual_group_of

    # ---- 2. GATE (env-own pairwise base-stock; searched + held-out re-scored) ---
    search_paths = base.make_paths(int(budget["search_paths"]), base.SEARCH_SEED)
    holdout_paths = base.make_paths(int(budget["holdout_paths"]), base.HOLDOUT_SEED)
    heuristic = base.search_best_pairwise_base_stock(search_paths, holdout_paths, budget["grid"])
    gate_cost = float(heuristic["holdout_mean_cost"])
    gate_oul = [int(x) for x in heuristic["oul_levels"]]  # integer OUL per supply relation

    # ---- 3. ANCHOR (gen-0 == gate guarantee; HARD assert for the residual head) ---
    # The residual head reproduces the gate byte-exact at the all-zero warm start. We
    # verify it end-to-end and treat a mismatch as a HARNESS_FAILURE (no silent fallback).
    anchor_cost = None
    if action_mode == "residual_base_stock":
        n = int(base._flat_param_count(depth, leaf))
        zeros = [0.0] * n
        anchor_mean, _anchor_se = base.soft_tree_cost_on_paths(
            zeros, depth, leaf, split, temp, holdout_paths,
            action_mode="residual_base_stock",
            backbone_levels=gate_oul,
            residual_group_of=group_of,
        )
        anchor_cost = float(anchor_mean)
        if abs(anchor_cost - gate_cost) >= ANCHOR_TOL:
            raise RuntimeError(
                "HARNESS_FAILURE: residual_base_stock gate-invertibility broken: "
                f"anchor_cost={anchor_cost!r} != gate_cost={gate_cost!r} "
                f"(|delta|={abs(anchor_cost - gate_cost):.3e} >= {ANCHOR_TOL}). "
                "gen-0 must equal the gate byte-exact at the zero warm start (Delta=0); "
                "the residual backbone wiring is broken."
            )

    # ---- 4. WARM START (the compiler resolves the CMA x0) ------------------------
    if compiled.warm_flat is not None:
        x0 = np.asarray(compiled.warm_flat, dtype=np.float64)
    else:
        # warm_start == "none": start CMA-ES from the model default (zeros). The honest
        # deploy floor still pins deployment at the gate via the anchor/gate candidates.
        x0 = np.zeros(int(base._flat_param_count(depth, leaf)), dtype=np.float64)

    # ---- 5. INNER CMA-ES per optimizer seed -------------------------------------
    per_seed_trained = []   # raw trained-xbest held-out cost (for the deployed_cost floor)
    per_seed_deployed = []  # honest per-seed floor min(trained, anchor, gate)
    per_seed_detail = []
    for s in range(int(n_seeds)):
        seed = SEED_BASE + SEED_STEP * s
        res = _train_one_seed(compiled, budget, generations, gate_oul, x0, seed, sigma)
        trained_cost, trained_se = base.soft_tree_cost_on_paths(
            res["best_flat"], depth, leaf, split, temp, holdout_paths,
            action_mode=action_mode,
            backbone_levels=gate_oul,
            residual_group_of=group_of,
        )
        trained_cost = float(trained_cost)
        per_seed_trained.append(trained_cost)
        # Honest per-seed floor: argmin over real-rollout costs {trained, anchor, gate}.
        floor_candidates = [trained_cost, gate_cost]
        if anchor_cost is not None:
            floor_candidates.append(anchor_cost)
        deployed = float(min(floor_candidates))
        per_seed_deployed.append(deployed)
        per_seed_detail.append({
            "seed": int(seed),
            "trained_holdout_mean": trained_cost,
            "trained_holdout_se": float(trained_se),
            "best_train_cost": float(res["best_train"]),
            "deployed_cost": deployed,
            "gap_pct": (deployed / gate_cost - 1.0) * 100.0,
            "train_seconds": float(res["train_seconds"]),
        })

    # ---- 6. ROBUST METRIC (README contract; keys + formula VERBATIM) ------------
    per_seed = [float(c) for c in per_seed_deployed]
    arr = np.asarray(per_seed, dtype=np.float64)
    mean_cost = float(arr.mean())
    std_cost = float(arr.std())  # population std over seeds
    gate_gap_pct = (mean_cost - gate_cost) / gate_cost * 100.0
    n_below = int((arr < gate_cost).sum())
    robust_gate_beat = bool(n_below == len(arr) and (mean_cost + std_cost) < gate_cost)
    mean_trained = float(np.mean(per_seed_trained))
    deployed_cost = float(min(mean_trained, gate_cost))

    return {
        "compiled_ok": True,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "per_seed": per_seed,
        "n_seeds": int(len(per_seed)),
        "gate_cost": gate_cost,
        "gate_gap_pct": gate_gap_pct,
        "n_seeds_below_gate": n_below,
        "deployed_cost": deployed_cost,
        "robust_gate_beat": robust_gate_beat,
        "error": None,
        # --- provenance (not in the minimal schema, but honest + useful to the loop) ---
        "problem": PROBLEM,
        "reference": "pirhooshyaran2021_mixed_scn_fig1_table5",
        "literature_verified": False,
        "baseline_kind": (
            "env_own_best_pairwise_base_stock (gate-beat; PADN has NO published DRL/PPO "
            "baseline -- gate comparator ONLY)"
        ),
        "backbone": compiled.backbone,
        "policy_action_mode": action_mode,
        "depth": depth,
        "leaf_type": leaf,
        "split_type": split,
        "temperature": temp,
        "per_echelon": compiled.per_echelon,
        "residual_group_of": group_of,
        "warm_start": compiled.warm_start_mode,
        "warm_started": compiled.warm_started,
        "n_params": compiled.n_params,
        "anchor_cost": anchor_cost,
        "mean_trained_cost": mean_trained,
        "gate_oul_levels": gate_oul,
        "gate_echelon_levels": [int(x) for x in heuristic["echelon_levels"]],
        "per_seed_detail": per_seed_detail,
        "budget": budget_name,
        "sigma_init": float(sigma),
        "cma_generations": int(generations),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PADN oracle CLI: score one policy-spec (DSL JSON) vs the env-own "
        "pairwise base-stock gate (no published DRL baseline; gate-beat only)."
    )
    p.add_argument("--spec", required=True, help="path to the policy-spec JSON")
    p.add_argument(
        "--problem",
        default=PROBLEM,
        help=f"must be {PROBLEM} (the only supported PADN oracle problem)",
    )
    p.add_argument(
        "--instance",
        type=int,
        default=0,
        help="accepted-and-IGNORED (PADN is a single fixed instance; present so the "
        "agent's fixed OWMR-shaped --instance call works unchanged)",
    )
    p.add_argument("--seeds", type=int, default=5, help="number of optimizer seeds (>=5, mandate)")
    p.add_argument("--budget", choices=sorted(base.BUDGETS), default="smoke")
    p.add_argument("--sigma_init", type=float, default=None,
                   help="optional CMA-ES sigma override (else the budget default)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--output_json",
        default=None,
        help="optional path to also write the result JSON (stdout is authoritative)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- harness-level preconditions (these are NOT spec errors) -> exit != 0 ----
    if args.problem != PROBLEM:
        print(
            json.dumps(_empty_result(
                f"unsupported --problem {args.problem!r}; this oracle only serves {PROBLEM}"
            )),
            flush=True,
        )
        return 2
    if int(args.seeds) < 5:
        print(
            json.dumps(_empty_result(
                f"--seeds={args.seeds} violates the >=5-seed seed-robust mandate"
            )),
            flush=True,
        )
        return 2

    # --instance is parsed but intentionally IGNORED (single fixed PADN instance).
    _ = bounded_worker_count(args.workers)

    # STDOUT IS THE JSON-ONLY CONTRACT CHANNEL. The inner CMA-ES / rollout routines log
    # via print()/sys.stdout, so we redirect Python-level stdout to stderr for the entire
    # computation; the result JSON is emitted to the RESTORED real stdout below (the Rust
    # action parses ALL of stdout, so any stray write would fail loudly, not be tolerated).
    real_stdout = sys.stdout
    rc = 0
    result = None
    sys.stdout = sys.stderr
    try:
        try:
            spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        except Exception as exc:
            # A spec file that does not parse is a compile failure, not a harness crash.
            result = _empty_result(f"spec load/parse failed: {exc.__class__.__name__}: {exc}")
        else:
            t0 = time.time()
            try:
                result = evaluate_spec(
                    spec=spec,
                    n_seeds=int(args.seeds),
                    budget_name=args.budget,
                    sigma_init=args.sigma_init,
                )
                result["eval_seconds"] = round(time.time() - t0, 2)
            except Exception as exc:
                # Genuine harness failure (broken gate-invertibility, missing binding, etc.):
                # surface the RAW traceback to stderr and exit non-zero (NO silent fallback).
                sys.stderr.write(traceback.format_exc())
                result = _empty_result(f"HARNESS_FAILURE: {exc.__class__.__name__}: {exc}")
                rc = 1
    finally:
        sys.stdout = real_stdout

    payload = json.dumps(result)
    print(payload, flush=True)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
