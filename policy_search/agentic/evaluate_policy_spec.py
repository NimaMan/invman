#!/usr/bin/env python
# =============================================================================
# evaluate_policy_spec.py -- the Python ORACLE CLI for policy_search/agentic
# =============================================================================
# OBJECTIVE
#   Score ONE policy-spec proposal (the README DSL JSON Codex emits) on a target
#   OWMR Kaynov instance under the lab's honest, seed-robust gate-beat metric, and
#   print the README evaluate-I/O JSON to stdout. This is step `evaluate_policy_spec`
#   of the crate's outer loop: it is invoked as a subprocess by the beden agent
#   (omurga action) once per generation and its stdout JSON is recorded in the
#   archive. It NEVER compares head-to-head against the published PPO scalar; the
#   only comparator is the in-repo echelon-base-stock GATE.
#
#   CLI:
#     python evaluate_policy_spec.py --spec spec.json \
#         --problem one_warehouse_multi_retailer --instance 14 --seeds 5 --budget small
#
# FULL ALGORITHMIC DESCRIPTION (matches the README "evaluate_policy_spec(spec)")
#   0. parse args; resolve instance N -> reference "kaynov2024_instance_N" via
#      common.get_reference. Hard CPU cap applied from argv before numpy/Rust import.
#   1. COMPILE  : spec(JSON) -> CompiledPolicy via policy_spec_compiler. A bad spec
#                 (unknown enum / wrong problem / wrong instance / build error) does
#                 NOT crash the harness: we return {compiled_ok:false, error:<raw>}
#                 with EXIT 0 (the loop uses the error as evidence). Only a genuine
#                 harness failure (e.g. instance missing, Rust binding absent) exits
#                 non-zero.
#   2. GATE     : grid-search the echelon-base-stock gate ONCE (parallel, <=4 workers,
#                 cached on disk by instance+budget+alloc-set) on the SEARCH CRN block,
#                 re-score each allocation argmin on the HELD-OUT CRN block, take the
#                 better allocation. This is the EXACT gate the production runner uses
#                 (reused via run_asymmetric_learned_vs_gate._search_gate_parallel).
#                 gate_cost = held-out mean of the best-allocation gate.
#   3. WARM     : attach the gate-invertible warm start so inner-CMA-ES generation-0
#                 reproduces the gate (target heads only; direct_orders unwarmed).
#   4. INNER OPT: for EACH of the >=5 optimizer seeds, run CMA-ES (small/screening
#                 budget for the MVP) on the Rust population-rollout oracle with the
#                 warm-started x0. Evaluate the trained xbest on the SAME held-out
#                 block (paired CRN) under each supported allocation; the per-seed
#                 learned cost is the better allocation of the better of
#                 {trained xbest, warm-start anchor} -- the honest deploy floor per
#                 seed (we never let a degenerate CMA seed report worse than the gate
#                 anchor it started from).
#   5. ROBUST METRIC (the README contract, computed structurally):
#                 per_seed              = [deployed-per-seed cost] (len == n_seeds)
#                 mean_cost / std_cost  = mean / population std of per_seed
#                 gate_cost             = held-out best-allocation gate cost
#                 gate_gap_pct          = (mean_cost - gate_cost)/gate_cost*100
#                 n_seeds_below_gate    = #{s : per_seed[s] < gate_cost}
#                 robust_gate_beat      = (n_seeds_below_gate == n_seeds) AND
#                                         (mean_cost + std_cost < gate_cost)
#                 deployed_cost         = min(mean trained-xbest cost, gate_cost)
#                                         == honest deploy floor (never below gate on
#                                            the strength of a lucky seed)
#   6. PRINT the result JSON to stdout (one object, the README schema) and exit 0.
#
# HONEST REPORTING (lab mandate, structural here)
#   - >=5 seeds always (the CLI rejects --seeds < 5 as a harness error, exit != 0).
#   - paired CRN: gate and every learned seed are scored on the identical held-out
#     demand-path block with the identical allocation-RNG anchors.
#   - robust_gate_beat is the conjunction above; anything weaker is reported as
#     robust_gate_beat=false (parity / not robust), never a "win".
#   - deployed_cost is the gate floor; a spec can never deploy below the gate.
#   - PPO is NOT in the schema and is NOT compared. (Cross-protocol context only.)
#
# BUDGETS (the inner CMA-ES screening sizes; keep the MVP smoke fast)
#   tiny / small are screening sizes (few generations, small population, a small
#   gate-search path block). full mirrors the production runner. The held-out
#   evaluation block size scales with the budget; the gate ARGMIN is stable with a
#   small search block (smooth base-stock cost surface) but the held-out re-score
#   uses the budget's holdout_paths so the reported gate/learned costs are honest.
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# --- hard CPU cap BEFORE numpy / Rust import (lab convention) --------------- #
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_OWMR_SCRIPT_DIR = _PACKAGE_ROOT / "scripts" / "one_warehouse_multi_retailer"
_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_PACKAGE_ROOT), str(_OWMR_SCRIPT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from invman.cpu_limits import (  # noqa: E402
    bounded_worker_count,
    configure_process_cpu_limits_from_argv,
)

configure_process_cpu_limits_from_argv(sys.argv[1:], default=4)

import numpy as np  # noqa: E402


# --------------------------------------------------------------------------- #
# Inner-CMA-ES budgets (screening sizes for the agentic loop / smoke).        #
#   training_episodes : CMA-ES generations
#   es_population      : CMA-ES population per generation
#   train_seed_batch   : per-candidate training demand seeds (averaged)
#   gate_search_paths  : CRN paths for the gate grid ARGMIN (cached)
#   holdout_paths      : CRN paths for the honest held-out re-score
# --------------------------------------------------------------------------- #
BUDGETS = {
    "tiny": {
        "training_episodes": 8,
        "es_population": 8,
        "train_seed_batch": 2,
        "gate_search_paths": 12,
        "holdout_paths": 128,
    },
    "small": {
        "training_episodes": 40,
        "es_population": 16,
        "train_seed_batch": 4,
        "gate_search_paths": 24,
        "holdout_paths": 512,
    },
    "full": {
        "training_episodes": 600,
        "es_population": 32,
        "train_seed_batch": 12,
        "gate_search_paths": 64,
        "holdout_paths": 4096,
    },
}


def _empty_result(error: str) -> dict:
    """README evaluate-I/O object for a spec that failed to compile/run."""
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


def _load_spec(spec_path: str) -> dict:
    text = Path(spec_path).read_text(encoding="utf-8")
    return json.loads(text)


def _training_namespace(reference, budget, compiled, seed, sigma_init, out_root):
    """A train()-compatible args namespace (mirrors the production OWMR runner)."""
    run_name = (
        f"aps_{reference['name']}_{compiled.policy_action_mode}_{compiled.leaf_type}"
        f"_d{compiled.depth}_{compiled.split_type}_t{compiled.temperature:g}"
        f"_{compiled.policy_state_mode}_pop{budget['es_population']}"
        f"_gen{budget['training_episodes']}_seed{seed}"
    )
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(sigma_init),
        es_population=int(budget["es_population"]),
        training_episodes=int(budget["training_episodes"]),
        mp_num_processors=1,  # parallelism is rayon inside Rust; no python pool
        save_every=max(1, int(budget["training_episodes"])),
        save_solutions=False,
        horizon=int(reference["benchmark_periods"]),
        seed=int(seed),
        train_seed_batch=int(budget["train_seed_batch"]),
        experiment_name=run_name,
        log_dir=str(out_root / "logs"),
        trained_models_dir=str(out_root / "models"),
    )


def _gate_cost_and_levels(reference, budget, workers, out_root, eval_allocations):
    """Search (cached) + held-out re-score the echelon-base-stock gate; return the
    best-allocation (cost, W, [r..], allocation, held-out cost array per allocation).
    Reuses the production runner's parallel gate search verbatim."""
    import run_asymmetric_learned_vs_gate as R
    from benchmark_learned_vs_heuristic import _heuristic_on_paths, _sample_demand_paths

    n_gate_search = int(budget["gate_search_paths"])
    search_paths = _sample_demand_paths(reference, n_gate_search, R.SEARCH_SEED_START)
    holdout_paths = _sample_demand_paths(reference, int(budget["holdout_paths"]), R.HOLDOUT_SEED_START)

    cache_dir = out_root / "gate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    allocs = tuple(sorted(eval_allocations))
    cache_path = cache_dir / f"{reference['name']}_gs{n_gate_search}_{'-'.join(allocs)}.json"
    if cache_path.exists():
        gate_searched = {
            a: {
                "warehouse_base_stock_level": int(v["warehouse_base_stock_level"]),
                "retailer_base_stock_levels": [int(x) for x in v["retailer_base_stock_levels"]],
                "search_mean_cost": float(v["search_mean_cost"]),
            }
            for a, v in json.loads(cache_path.read_text()).items()
        }
    else:
        gate_searched = R._search_gate_parallel(
            reference, list(eval_allocations), search_paths, workers
        )
        cache_path.write_text(json.dumps(gate_searched, default=float, indent=2))

    gate = {}
    for allocation, g in gate_searched.items():
        holdout_costs = _heuristic_on_paths(
            reference,
            g["warehouse_base_stock_level"],
            g["retailer_base_stock_levels"],
            allocation,
            holdout_paths,
            R.ALLOC_SEED_HOLDOUT,
        )
        gate[allocation] = {
            **g,
            "holdout_costs": holdout_costs,
            "holdout_mean": float(holdout_costs.mean()),
        }
    best_alloc = min(gate, key=lambda a: gate[a]["holdout_mean"])
    return gate, best_alloc, holdout_paths


def _eval_policy_on_holdout(reference, model, flat, action_mode, allocations, holdout_paths):
    """Mean held-out cost of `flat` under each allocation; return {alloc: mean}."""
    from benchmark_learned_vs_heuristic import _soft_tree_on_paths
    import run_asymmetric_learned_vs_gate as R

    out = {}
    for allocation in allocations:
        costs = _soft_tree_on_paths(
            reference, model, flat, allocation, action_mode, holdout_paths, R.ALLOC_SEED_HOLDOUT
        )
        out[allocation] = float(costs.mean())
    return out


def evaluate_spec(
    spec: dict,
    instance: int,
    n_seeds: int,
    budget_name: str,
    sigma_init: float,
    workers: int,
) -> dict:
    """Compile, train (>=n_seeds), evaluate seed-robustly vs the gate, return the
    README evaluate-I/O dict. Raises only on a genuine harness failure (caught by
    main and turned into a non-zero exit); a non-compiling spec returns an
    _empty_result with compiled_ok=false (exit 0)."""
    import common  # OWMR reference + builders
    from invman.es_mp import train
    from policy_spec_compiler import (
        PolicySpecError,
        attach_gate_warm_start,
        compile_policy_spec,
    )
    from benchmark_learned_vs_heuristic import (
        _get_model_fitness,
        _get_population_fitness,
    )

    instance_name = f"kaynov2024_instance_{int(instance)}"
    reference = common.get_reference(instance_name)  # raises if instance missing (harness error)
    budget = BUDGETS[budget_name]
    out_root = _THIS_DIR / "outputs" / "evaluate_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. COMPILE (spec errors => compiled_ok=false, NOT a harness failure) ---
    try:
        compiled = compile_policy_spec(spec, reference)
    except PolicySpecError as exc:
        return _empty_result(f"PolicySpecError: {exc}")

    eval_allocs = compiled.eval_allocations

    # ---- 2. GATE (searched once, cached, re-scored on the held-out block) -------
    gate, gate_best_alloc, holdout_paths = _gate_cost_and_levels(
        reference, budget, workers, out_root, eval_allocs
    )
    gate_best = gate[gate_best_alloc]
    gate_cost = float(gate_best["holdout_mean"])

    # ---- 3. WARM START (gate-invertible anchor; target heads only) --------------
    warm_mode = spec.get("warm_start", "gate_invertible")
    compiled = attach_gate_warm_start(
        compiled,
        reference,
        warm_mode,
        gate_best["warehouse_base_stock_level"],
        gate_best["retailer_base_stock_levels"],
    )

    # Anchor held-out cost (same for every seed: the warm start is deterministic).
    anchor_cost = None
    if compiled.warm_started and compiled.warm_flat is not None:
        anchor_alloc_costs = _eval_policy_on_holdout(
            reference, compiled.model, compiled.warm_flat,
            compiled.policy_action_mode, eval_allocs, holdout_paths,
        )
        anchor_cost = min(anchor_alloc_costs.values())

    # ---- 4. INNER CMA-ES per optimizer seed -------------------------------------
    # train_allocation: use the gate's best allocation as the training rationing rule
    # so the inner optimization is consistent with the protocol the gate was tuned on.
    train_allocation = gate_best_alloc if gate_best_alloc in eval_allocs else eval_allocs[0]

    per_seed_trained = []  # raw trained-xbest cost (for deployed_cost floor)
    per_seed_deployed = []  # min(trained, anchor) per seed (honest per-seed floor)
    seed_base = 700_000
    for s in range(int(n_seeds)):
        seed = seed_base + 101 * s
        model = common.build_soft_tree_model(
            reference,
            depth=compiled.depth,
            temperature=compiled.temperature,
            split_type=compiled.split_type,
            leaf_type=compiled.leaf_type,
            policy_action_mode=compiled.policy_action_mode,
            policy_state_mode=compiled.policy_state_mode,
        )
        train_args = _training_namespace(reference, budget, compiled, seed, sigma_init, out_root)
        if compiled.warm_flat is not None:
            train_args.cma_x0 = compiled.warm_flat

        trained_model, _fitness_hist = train(
            model=model,
            get_model_fitness=_get_model_fitness(
                model, reference, train_allocation, compiled.policy_action_mode
            ),
            get_population_fitness=_get_population_fitness(
                model, reference, train_allocation, compiled.policy_action_mode
            ),
            args=train_args,
            same_seed=True,  # CRN within each CMA-ES population batch (low-variance)
        )
        trained_flat = np.asarray(
            trained_model.get_model_flat_params(), dtype=np.float32
        ).tolist()
        trained_alloc_costs = _eval_policy_on_holdout(
            reference, trained_model, trained_flat,
            compiled.policy_action_mode, eval_allocs, holdout_paths,
        )
        trained_cost = min(trained_alloc_costs.values())
        per_seed_trained.append(float(trained_cost))
        # Honest per-seed floor: never report worse than the gate-reproducing anchor.
        deployed = trained_cost if anchor_cost is None else min(trained_cost, anchor_cost)
        per_seed_deployed.append(float(deployed))

    # ---- 5. ROBUST METRIC (README contract) -------------------------------------
    per_seed = [float(c) for c in per_seed_deployed]
    arr = np.asarray(per_seed, dtype=np.float64)
    mean_cost = float(arr.mean())
    std_cost = float(arr.std())  # population std over seeds
    gate_gap_pct = (mean_cost - gate_cost) / gate_cost * 100.0
    n_below = int((arr < gate_cost).sum())
    robust_gate_beat = bool(n_below == len(arr) and (mean_cost + std_cost) < gate_cost)
    # deployed_cost = honest deploy floor: better of {mean trained xbest, gate}.
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
        "instance": instance_name,
        "policy_action_mode": compiled.policy_action_mode,
        "policy_state_mode": compiled.policy_state_mode,
        "backbone": compiled.backbone,
        "depth": compiled.depth,
        "leaf_type": compiled.leaf_type,
        "split_type": compiled.split_type,
        "temperature": compiled.temperature,
        "warm_started": compiled.warm_started,
        "anchor_cost": anchor_cost,
        "mean_trained_cost": mean_trained,
        "gate_best_allocation": gate_best_alloc,
        "gate_warehouse_level": int(gate_best["warehouse_base_stock_level"]),
        "gate_retailer_levels": [int(v) for v in gate_best["retailer_base_stock_levels"]],
        "train_allocation": train_allocation,
        "budget": budget_name,
        "sigma_init": float(sigma_init),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Oracle CLI: score one policy-spec (DSL JSON) vs the OWMR gate."
    )
    p.add_argument("--spec", required=True, help="path to the policy-spec JSON")
    p.add_argument(
        "--problem",
        default="one_warehouse_multi_retailer",
        help="must be one_warehouse_multi_retailer (the only supported oracle problem)",
    )
    p.add_argument("--instance", type=int, default=14, help="Kaynov instance number (e.g. 14)")
    p.add_argument("--seeds", type=int, default=5, help="number of optimizer seeds (>=5, mandate)")
    p.add_argument("--budget", choices=sorted(BUDGETS), default="small")
    p.add_argument("--sigma_init", type=float, default=0.10)
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
    if args.problem != "one_warehouse_multi_retailer":
        print(
            json.dumps(_empty_result(
                f"unsupported --problem {args.problem!r}; oracle only serves "
                "one_warehouse_multi_retailer"
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

    workers = bounded_worker_count(args.workers)

    # STDOUT IS THE JSON-ONLY CONTRACT CHANNEL (the beden agent parses the whole of
    # stdout as the evaluate-I/O JSON). The inner CMA-ES / pycma training routines log
    # via print()/sys.stdout, so we redirect Python-level stdout to stderr for the
    # entire computation; the result JSON is emitted to the RESTORED real stdout below.
    # (The Rust action deliberately stays strict and parses all of stdout, so any future
    # stray stdout write fails loudly rather than being silently tolerated.)
    real_stdout = sys.stdout
    rc = 0
    result = None
    sys.stdout = sys.stderr
    try:
        try:
            spec = _load_spec(args.spec)
        except Exception as exc:
            # A spec file that does not parse is a compile failure, not a harness crash.
            result = _empty_result(
                f"spec load/parse failed: {exc.__class__.__name__}: {exc}"
            )
        else:
            t0 = time.time()
            try:
                result = evaluate_spec(
                    spec=spec,
                    instance=int(args.instance),
                    n_seeds=int(args.seeds),
                    budget_name=args.budget,
                    sigma_init=float(args.sigma_init),
                    workers=workers,
                )
                result["eval_seconds"] = round(time.time() - t0, 2)
            except Exception as exc:
                # Genuine harness failure (missing instance, broken Rust binding, etc.):
                # surface the RAW traceback to stderr and exit non-zero (no silent fallback).
                sys.stderr.write(traceback.format_exc())
                result = _empty_result(
                    f"HARNESS_FAILURE: {exc.__class__.__name__}: {exc}"
                )
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
