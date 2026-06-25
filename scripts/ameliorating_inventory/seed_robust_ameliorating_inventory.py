"""
Seed-ROBUST learned-vs-gate runner for the AMELIORATING INVENTORY family
(faithful Pahr & Grunow 2025 average-PROFIT blending env,
src/problems/ameliorating_inventory/average_profit_blending_env.rs).

OBJECTIVE
---------
The existing single-seed runner (autoresearch_ameliorating_inventory_average_profit.py)
reports ONE CMA-ES optimizer seed per instance. Per the project mandate (centralized in
invman/optimizer_seed_robustness_policy.py, problem_id "ameliorating_inventory",
mode="seeds"), a headline learned-vs-gate result must be a MEAN +/- SAMPLE-STD over
>= 5 independent optimizer seeds, never a single seed. This runner re-runs the EXACT
single-seed training protocol (no env / policy / Rust changes; the existing module is
imported and its helpers reused verbatim) once per optimizer seed and aggregates with
the shared srp summary/verdict logic.

PROFIT SIGN CONVENTION (read before touching the numbers)
---------------------------------------------------------
This problem is PROFIT-MAXIMIZING (long-run average profit), while srp's standardized
summary assumes lower-is-better "costs". Two measures keep the semantics honest:
  1. The per-seed records feed srp NEGATED profits:
         gate_cost          = -gate_profit
         best_learned_cost  = -learned_profit
     so the standardized keys learned_seed_mean / gate_seed_mean (and stds) are means
     of NEGATED profits (lower = better, i.e. more profit). Profit-oriented convenience
     keys (learned_profit_seed_mean/std, gate_profit_seed_mean/std) are written
     alongside so no reader has to flip signs mentally.
  2. savings_pct_vs_gate is PRECOMPUTED per seed with the correct profit orientation:
         savings_pct_vsgate = 100 * (learned_profit - gate_profit) / abs(gate_profit)
     (positive = learned beats the gate; abs() keeps the sign right even for a
     negative-profit gate). srp.build_seed_robust_summary uses a precomputed
     savings_pct_vs_gate when present in every record (verified in its source), so the
     cost-style derivation 100*(gate-learned)/gate -- which would be sign-WRONG on
     negated profits -- is never applied.

GATE SEMANTICS
--------------
The same-protocol gate is the single-seed runner's own heuristic gate: the best
order-up-to purchase level tuned on that seed's held-out eval block
(tune_order_up_to), evaluated on the full eval block under the same paired CRN. The
gate is re-tuned PER OPTIMIZER SEED on that seed's eval block (the eval seeds are
derived from the optimizer seed exactly as in the single-seed runner), so each seed's
learned-vs-gate margin is paired. The published perfect-information LP upper bound is
recorded as CONTEXT ONLY (gap_to_bound_pct per seed); it is a loose bound, never a
same-protocol comparator. NOTE the candidate set deployed on the eval block includes
the order-up-to anchor itself (the existing runner's honest floor), so per-seed
savings is >= 0 by construction; the verdict still distinguishes ROBUST_BEAT
(every seed strictly above the gate by more than the cross-seed std) from PARITY.

ALGORITHM (full description)
----------------------------
1. Import the existing single-seed module via importlib (exemplar pattern from
   scripts/multi_echelon/seed_robust_divergent_multi_echelon.py); reuse its
   parse_dataset / rollout_kwargs / order_up_to_warm_start / tune_order_up_to /
   evaluate / train / BUDGETS verbatim.
2. For each optimizer seed s (default = srp canonical 9001..9005):
   a. eval_seeds = [s + 1_000_000 + i] (the single-seed runner's own derivation).
   b. GATE: tune the order-up-to level on the eval block, evaluate at full eval reps
      -> gate_profit (same-protocol, paired, at this seed).
   c. TRAIN: CMA-ES (seeded s) from the order-up-to warm start on the train block.
   d. DEPLOY: evaluate the candidate set {order_up_to_anchor, cma xbest, cma
      xfavorite (deploy_endpoint="floor" default), best per-generation incumbent} on
      the held-out eval block; learned_profit = best candidate (the single-seed
      runner's exact selection rule).
   e. Record gate/learned profits, negated-profit cost keys, precomputed
      savings_pct_vs_gate, gap-to-LP-bound context, endpoint diagnostics.
3. Aggregate with srp.run_over_seeds (shared >= 5-seed enforcement, sample std,
   shared verdict rule) and write one JSON:
     real run  -> outputs/ameliorating_inventory/seed_robust_report.json
                  (non-default instance -> seed_robust_report_<instance>.json)
     --smoke   -> outputs/ameliorating_inventory/smoke_seed_robust/... ONLY; a smoke
                  run can NEVER write the real artifact path. --smoke also forces the
                  smallest existing budget preset ("smoke" in the single-seed
                  runner's BUDGETS) and mp_num_processors=1.

CPU CAP / USAGE
---------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/ameliorating_inventory/seed_robust_ameliorating_inventory.py \
      --instance spirits_0001 --budget full \
      --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2

  Smoke test (tiny budget, 1 worker, separate smoke output path):
  python scripts/ameliorating_inventory/seed_robust_ameliorating_inventory.py --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import (  # noqa: E402
    configure_process_cpu_limits,
    configure_process_cpu_limits_from_argv,
)

# Must run before the Rust binding's Rayon pool first initializes. A --smoke run is
# re-capped to 1 worker after arg parsing (still before any rollout call).
configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np  # noqa: E402

from invman import optimizer_seed_robustness_policy as _srp  # noqa: E402

# Load the existing single-seed runner as a module and reuse its functions verbatim
# (exemplar pattern; importing it only defines helpers -- main() is __main__-guarded).
_RUNNER = (
    PACKAGE_ROOT
    / "scripts"
    / "ameliorating_inventory"
    / "autoresearch_ameliorating_inventory_average_profit.py"
)
_spec = importlib.util.spec_from_file_location(
    "autoresearch_ameliorating_inventory_average_profit", _RUNNER
)
_tmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tmp)

PROBLEM_ID = "ameliorating_inventory"
DEFAULT_INSTANCE = "spirits_0001"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", choices=sorted(_tmp.INSTANCES), default=DEFAULT_INSTANCE)
    p.add_argument("--budget", choices=sorted(_tmp.BUDGETS), default="full")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(_srp.seeds_for(PROBLEM_ID)),
                   help="Optimizer seeds (>= 5 distinct required; default canonical 9001..9005).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny smoke run: forces the 'smoke' budget preset, "
                        "mp_num_processors=1, and a separate smoke output path.")
    p.add_argument("--mp_num_processors", type=int, default=2)
    # Pass-through knobs, defaults identical to the single-seed runner.
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["linear"], default="linear")
    p.add_argument("--sigma_init", type=float, default=0.5)
    p.add_argument("--order_up_to_ceiling", type=float, default=25.0)
    p.add_argument("--deploy_endpoint", choices=["floor", "xbest", "xfavorite"],
                   default="floor",
                   help="Which trained CMA-ES endpoint(s) join the held-out best-of "
                        "set (same semantics as the single-seed runner; 'floor' adds "
                        "the distribution-mean xfavorite to xbest, downside-safe).")
    return p.parse_args()


def train_one_seed(parsed, budget: dict, seed: int) -> dict:
    """One full single-seed-protocol run (gate + train + deploy) at this optimizer seed.

    Mirrors autoresearch_ameliorating_inventory_average_profit.main() exactly, minus
    the JSON/TSV bookkeeping; returns the per-seed record srp aggregates.
    """
    data = _tmp.parse_dataset(parsed.instance)
    env = data["env"]
    bound = float(data["published_bound"])
    n_eval = int(budget["eval_seeds"])
    eval_seeds = [seed + 1_000_000 + i for i in range(n_eval)]

    kw_train = _tmp.rollout_kwargs(env, parsed.depth, parsed.temperature, parsed.split_type,
                                   parsed.leaf_type, budget["train_periods"], budget["warm_up"])
    kw_eval = _tmp.rollout_kwargs(env, parsed.depth, parsed.temperature, parsed.split_type,
                                  parsed.leaf_type, budget["eval_periods"], budget["eval_warm_up"])

    t0 = time.time()
    # --- same-protocol gate at THIS seed: best order-up-to level on the eval block ---
    tune = _tmp.tune_order_up_to(env, kw_eval, parsed.depth, eval_seeds,
                                 parsed.order_up_to_ceiling)
    gate_level = int(tune["best_level"])
    x0 = _tmp.order_up_to_warm_start(env, parsed.depth, gate_level)
    gate = _tmp.evaluate(kw_eval, x0, eval_seeds)
    gate_profit = float(gate["mean_profit"])

    # --- CMA-ES (maximizing profit), warm-started at the gate's order-up-to encoding ---
    cma_best, cma_xfavorite, gen_candidates, history = _tmp.train(
        kw_train, x0, budget["popsize"], budget["generations"], parsed.sigma_init, seed,
    )

    # --- held-out best-of deployment (single-seed runner's exact selection rule) ---
    candidates = {"order_up_to_anchor": x0}
    if parsed.deploy_endpoint in ("floor", "xbest"):
        candidates["cma_incumbent"] = cma_best
    if parsed.deploy_endpoint in ("floor", "xfavorite"):
        candidates["cma_xfavorite"] = cma_xfavorite
    if history:
        sub = eval_seeds[: max(4, n_eval // 4)]
        best_gen_idx = int(np.argmax([
            _tmp.evaluate(kw_eval, c, sub)["mean_profit"] for c in gen_candidates
        ]))
        candidates[f"gen_best@{best_gen_idx}"] = gen_candidates[best_gen_idx]

    cand_evals = {name: _tmp.evaluate(kw_eval, p, eval_seeds) for name, p in candidates.items()}
    learned_source = max(cand_evals, key=lambda k: cand_evals[k]["mean_profit"])
    learned = cand_evals[learned_source]
    learned_profit = float(learned["mean_profit"])

    # PROFIT-oriented savings (positive = learned beats gate); precomputed so srp never
    # applies its cost-style formula to the negated profits below.
    savings_pct = 100.0 * (learned_profit - gate_profit) / abs(gate_profit)
    record = {
        "seed": seed,
        # srp keys (NEGATED profits, lower-is-better; see module docstring):
        "gate_cost": -gate_profit,
        "best_learned_cost": -learned_profit,
        "savings_pct_vs_gate": savings_pct,
        # profit-oriented fields (the readable truth):
        "gate_profit": gate_profit,
        "gate_sem": float(gate["sem_profit"]),
        "gate_order_up_to_level": gate_level,
        "learned_profit": learned_profit,
        "learned_sem": float(learned["sem_profit"]),
        "learned_source": learned_source,
        "xbest_profit": (float(cand_evals["cma_incumbent"]["mean_profit"])
                         if "cma_incumbent" in cand_evals else None),
        "xfavorite_profit": (float(cand_evals["cma_xfavorite"]["mean_profit"])
                             if "cma_xfavorite" in cand_evals else None),
        # context only -- the LP bound is loose, never a same-protocol comparator:
        "perfect_information_lp_bound_CONTEXT_ONLY": bound,
        "gap_to_bound_pct_CONTEXT_ONLY": 100.0 * (bound - learned_profit) / bound,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[seed {seed}] gate {gate_profit:.2f} (S={gate_level})  "
          f"learned {learned_profit:.2f} ({learned_source})  "
          f"savings {savings_pct:+.2f}%  ({record['seconds']}s)")
    return record


def main():
    parsed = parse_args()
    if parsed.smoke:
        # Smallest existing budget preset + 1 worker; re-cap BEFORE any rollout call.
        parsed.budget = "smoke"
        parsed.mp_num_processors = 1
        configure_process_cpu_limits(1)
    budget = dict(_tmp.BUDGETS[parsed.budget])

    out_dir = PACKAGE_ROOT / "outputs" / "ameliorating_inventory"
    if parsed.smoke:
        # A smoke run can NEVER write the real artifact path.
        out_dir = out_dir / "smoke_seed_robust"
        json_path = out_dir / f"seed_robust_report_{parsed.instance}_smoke.json"
    elif parsed.instance == DEFAULT_INSTANCE:
        json_path = out_dir / "seed_robust_report.json"
    else:
        json_path = out_dir / f"seed_robust_report_{parsed.instance}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared seed loop + >= 5-seed enforcement + sample-std + verdict (srp is the
    # single source of truth; savings_pct_vs_gate is precomputed in every record, so
    # build_seed_robust_summary uses the profit-oriented values, never 100*(g-l)/g).
    result = _srp.run_over_seeds(
        PROBLEM_ID,
        lambda seed: train_one_seed(parsed, budget, seed),
        seeds=parsed.seeds,
    )
    per_seed = result["per_seed"]

    # Profit-oriented convenience summary (un-negated mirrors of the srp keys).
    learned_p = _srp.summarize_values([s["learned_profit"] for s in per_seed])
    gate_p = _srp.summarize_values([s["gate_profit"] for s in per_seed])

    out = {
        "family": PROBLEM_ID,
        "benchmark": "seed_robust_ameliorating_inventory",
        "model": "average_profit_blending_env (Pahr & Grunow 2025, faithful)",
        "objective_orientation": "PROFIT_MAXIMIZING (gate_cost/best_learned_cost keys are "
                                 "NEGATED profits; savings_pct_vs_gate is profit-oriented, "
                                 "positive = learned beats gate)",
        "commit": _tmp._git_short_commit(),
        "instance": parsed.instance,
        "budget": parsed.budget,
        "smoke": parsed.smoke,
        "config": {
            "depth": parsed.depth,
            "temperature": parsed.temperature,
            "split_type": parsed.split_type,
            "leaf_type": parsed.leaf_type,
            "sigma_init": parsed.sigma_init,
            "order_up_to_ceiling": parsed.order_up_to_ceiling,
            "deploy_endpoint": parsed.deploy_endpoint,
            "mp_num_processors": parsed.mp_num_processors,
            **budget,
        },
        "gate_semantics": "best order-up-to purchase level tuned per optimizer seed on "
                          "that seed's held-out eval block (same protocol, paired CRN)",
        "perfect_information_lp_bound_CONTEXT_ONLY":
            float(_tmp.INSTANCES[parsed.instance]["published_bound"]),
        # profit-oriented convenience summary:
        "learned_profit_seed_mean": learned_p["seed_mean"],
        "learned_profit_seed_std": learned_p["seed_std"],
        "gate_profit_seed_mean": gate_p["seed_mean"],
        "gate_profit_seed_std": gate_p["seed_std"],
        # seeds + per_seed + standardized srp summary keys (n_optimizer_seeds,
        # learned/gate_seed_mean/std on NEGATED profits, savings_pct_seed_mean/std,
        # frac_seeds_beating_gate, verdict_vs_same_protocol_gate):
        **result,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{parsed.instance}  budget={parsed.budget}  smoke={parsed.smoke}  "
          f"seeds={out['seeds']}")
    print(f"GATE profit seed-mean    {gate_p['seed_mean']:.2f} +/- {gate_p['seed_std']:.2f}")
    print(f"LEARNED profit seed-mean {learned_p['seed_mean']:.2f} +/- {learned_p['seed_std']:.2f}")
    print(f"SAVINGS vs gate  {out['savings_pct_seed_mean']:+.2f}% +/- "
          f"{out['savings_pct_seed_std']:.2f}%  (beating gate {out['frac_seeds_beating_gate']})")
    print(f"LP bound context only: {out['perfect_information_lp_bound_CONTEXT_ONLY']:.4f}")
    print(f"VERDICT (vs same-protocol gate): {out['verdict_vs_same_protocol_gate']}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
