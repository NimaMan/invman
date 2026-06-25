"""
Seed-ROBUST learned-vs-gate runner for general_backorder_fixed_cost
(Geevers, van Hezewijk & Mes 2024, CardBoard Company general network, set 1).

OBJECTIVE
---------
The paper's headline for this problem is a 5-seed claim ("learned soft tree
7,837.0 +/- 189.7, an improvement of 24.3% +/- 1.8% over the reproduced constant
node-base-stock benchmark 10,354.8, all 5 seeds below the benchmark"). Per the
project mandate, every such headline must be produced by a runner that loops
>= 5 independent CMA-ES optimizer seeds through the SAME training entry point
and reports the standardized cross-seed summary from
invman/optimizer_seed_robustness_policy.py (srp) -- never single-seed or
best-of-N, and never with per-script drift in std/verdict conventions. This
script is that runner: it reuses the EXISTING single-seed autoresearch entry
point's helpers verbatim (no new env, policy, or Rust code) and delegates the
seed loop + aggregation + verdict to srp.run_over_seeds.

GATE SEMANTICS (same-protocol comparator)
-----------------------------------------
gate_cost = the autoresearch script's keep/discard GATE: the IN-REPO
reproduction of the published constant node-base-stock benchmark, i.e.
multi_echelon_general_backorder_fixed_cost_simulate_base_stock at the published
levels, averaged over 3 fixed sim seeds (1234, 5678, 9012) x benchmark_replications
(500 for set 1) -> ~10,354.8 (published: 10,467). This gate does NOT depend on the
optimizer seed (it is a fixed policy simulated once), so gate_seed_std = 0 by
construction; the cross-seed variation lives entirely in the learned costs. The
warm-start (generation 0 == the same benchmark policy, evaluated on the exact
held-out CRN eval block) is recorded alongside as a protocol cross-check, and the
published PPO best (8,714) is carried as CROSS-PROTOCOL CONTEXT ONLY -- never a
head-to-head verdict.

ALGORITHM (full description)
----------------------------
1. importlib-load scripts/general_backorder_fixed_cost/
   autoresearch_general_backorder_fixed_cost.py and reuse its functions verbatim:
   BUDGETS, build_action_bounds, warm_start_flat_params, make_seed_block,
   population_costs, paired_eval (and its module-global REFERENCE_NAME contract).
2. Compute the gate ONCE (optimizer-seed independent): repo heuristic
   reproduction cost as above. With --smoke the gate uses min(replications, 20)
   per sim seed (an argument to the binding, not an env change) to stay tiny.
3. Evaluate the warm-start vector ONCE on the held-out CRN eval block (it is
   identical for every optimizer seed) -> warm_start_heldout cross-check.
4. train_one_seed(seed): CMA-ES (cma library) warm-started at the published-level
   encoding with small sigma, fitness = mean rollout cost over the FIXED train
   seed block (disjoint from eval), exactly as the single-seed entry point;
   deploy es.result.xbest; paired held-out eval on the eval block; return
   {seed, gate_cost, best_learned_cost, savings_pct_vs_gate (=100*(gate-learned)/gate,
   positive == learned better), learned_sem, best_train_cost, generations, seconds}.
5. srp.run_over_seeds("general_backorder_fixed_cost", train_one_seed, seeds=...)
   enforces >= 5 distinct optimizer seeds, aggregates with the shared sample-std
   (n-1) convention, and emits the standardized summary keys (n_optimizer_seeds,
   learned/gate_seed_mean/std, savings_pct_seed_mean/std, frac_seeds_beating_gate,
   verdict_vs_same_protocol_gate).
6. Write the JSON artifact:
     real run  -> outputs/general_backorder_fixed_cost/seed_robust_report.json
     --smoke   -> outputs/general_backorder_fixed_cost/smoke_seed_robust/
                  seed_robust_report_smoke.json  (NEVER the real path)
   --smoke also forces the entry point's existing tiny "smoke" budget preset
   (popsize 8, 8 generations, 4 train seeds, 64 eval seeds) and mp_num_processors 1.

CPU CAP / USAGE
---------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/general_backorder_fixed_cost/seed_robust_general_backorder.py \
      --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2

  smoke test (tiny budget, separate artifact path, 1 worker):
  python scripts/general_backorder_fixed_cost/seed_robust_general_backorder.py --smoke
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

from invman.cpu_limits import configure_process_cpu_limits_from_argv  # noqa: E402

# HARD CPU CAP -- must run BEFORE importing numpy/invman_rust. Smoke runs are
# pinned to 1 worker unless --mp_num_processors is given explicitly.
_DEFAULT_WORKERS = 1 if "--smoke" in sys.argv[1:] else 2
configure_process_cpu_limits_from_argv(sys.argv[1:], default=_DEFAULT_WORKERS)

import numpy as np  # noqa: E402,F401  (loaded after the CPU cap, used by the entry point)

import invman_rust as ir  # noqa: E402

from invman import optimizer_seed_robustness_policy as srp  # noqa: E402

PROBLEM_ID = "general_backorder_fixed_cost"

# Load the existing single-seed autoresearch entry point and reuse its helpers verbatim.
_RUNNER = PACKAGE_ROOT / "scripts" / "general_backorder_fixed_cost" / (
    "autoresearch_general_backorder_fixed_cost.py")
_spec = importlib.util.spec_from_file_location("autoresearch_general_backorder_fixed_cost", _RUNNER)
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

GATE_SIM_SEEDS = (1234, 5678, 9012)  # exactly the autoresearch script's gate sim seeds
SMOKE_GATE_REPLICATIONS = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-robust (>=5 optimizer seeds) learned-vs-gate "
                                            "runner for general_backorder_fixed_cost.")
    p.add_argument("--reference", default=_ar.DEFAULT_REFERENCE_NAME)
    p.add_argument("--budget", choices=sorted(_ar.BUDGETS), default="full")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Optimizer seeds (default: the canonical srp list 9001..9005; "
                        "seeds-mode requires >= 5 distinct).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny plumbing test: forces the 'smoke' budget preset, 1 worker, and a "
                        "SEPARATE smoke artifact path (never the real report).")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--leaf_type", choices=["constant", "linear"], default="constant")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--sigma_init", type=float, default=None,
                   help="Override the budget preset's sigma_init.")
    p.add_argument("--mp_num_processors", type=int, default=_DEFAULT_WORKERS)
    return p.parse_args()


def compute_gate_cost(reference: str, replications: int) -> float:
    """The autoresearch script's keep/discard gate: repo reproduction of the published
    constant node-base-stock benchmark (fixed policy => optimizer-seed independent)."""
    means = []
    for s in GATE_SIM_SEEDS:
        d = ir.multi_echelon_general_backorder_fixed_cost_simulate_base_stock(
            reference, None, int(replications), int(s), None)
        means.append(float(d["mean_cost"]))
    return float(sum(means) / len(means))


def main() -> None:
    parsed = parse_args()
    budget_name = "smoke" if parsed.smoke else parsed.budget
    budget = _ar.BUDGETS[budget_name]
    sigma_init = parsed.sigma_init if parsed.sigma_init is not None else budget["sigma_init"]
    seeds = srp.seeds_for(PROBLEM_ID, parsed.seeds)

    # The entry point's helpers read REFERENCE_NAME as a module global.
    _ar.REFERENCE_NAME = parsed.reference

    ref = dict(ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance(parsed.reference))
    levels = list(ref["benchmark_base_stock_levels"])
    published_benchmark = float(ref["published_benchmark_cost"])
    _ppo = ref["published_ppo_best_cost"]
    published_ppo_best = float(_ppo) if _ppo is not None else None

    min_values, max_values = _ar.build_action_bounds(ref["num_warehouses"], ref["num_retailers"])
    x0, input_dim = _ar.warm_start_flat_params(levels, min_values, max_values,
                                               parsed.depth, parsed.leaf_type)
    train_seeds = _ar.make_seed_block(_ar.TRAIN_SEED_BASE, _ar.TRAIN_SEED_STRIDE,
                                      budget["n_train_seeds"])
    eval_seeds = _ar.make_seed_block(_ar.EVAL_SEED_BASE, _ar.EVAL_SEED_STRIDE,
                                     budget["n_eval_seeds"])
    assert not (set(train_seeds) & set(eval_seeds)), "train/eval seed blocks must be disjoint"

    gate_replications = (min(int(ref["benchmark_replications"]), SMOKE_GATE_REPLICATIONS)
                         if parsed.smoke else int(ref["benchmark_replications"]))
    gate_cost = compute_gate_cost(parsed.reference, gate_replications)

    # Warm-start (gen 0 == the benchmark policy) on the exact eval block: identical for
    # every optimizer seed, so evaluate it ONCE as the protocol cross-check.
    warm_mean, warm_sem = _ar.paired_eval(x0, input_dim, parsed.depth, min_values, max_values,
                                          parsed.leaf_type, parsed.temperature, eval_seeds)
    print(f"[setup] reference={parsed.reference} budget={budget_name} depth={parsed.depth} "
          f"leaf={parsed.leaf_type} sigma={sigma_init} seeds={seeds}")
    print(f"[gate] repo heuristic reproduction {gate_cost:.1f} "
          f"({len(GATE_SIM_SEEDS)}x{gate_replications} reps; published {published_benchmark:.0f}); "
          f"warm-start held-out {warm_mean:.1f} +/- {warm_sem:.1f}")

    import cma

    def train_one_seed(seed: int) -> dict:
        t0 = time.time()
        es = cma.CMAEvolutionStrategy(
            [float(x) for x in x0], float(sigma_init),
            {"popsize": budget["popsize"], "seed": int(seed),
             "maxiter": budget["generations"], "verbose": -9},
        )
        gen = 0
        best_train = float("inf")
        while not es.stop():
            solutions = es.ask()
            fitness = _ar.population_costs(solutions, input_dim, parsed.depth, min_values,
                                           max_values, parsed.leaf_type, parsed.temperature,
                                           train_seeds)
            es.tell(solutions, fitness.tolist())
            gen += 1
            best_train = min(best_train, float(fitness.min()))
        learned_mean, learned_sem = _ar.paired_eval(es.result.xbest, input_dim, parsed.depth,
                                                    min_values, max_values, parsed.leaf_type,
                                                    parsed.temperature, eval_seeds)
        savings = 100.0 * (gate_cost - learned_mean) / gate_cost
        rec = {
            "seed": int(seed),
            "gate_cost": gate_cost,
            "best_learned_cost": float(learned_mean),
            "learned_sem": float(learned_sem),
            "savings_pct_vs_gate": savings,
            "best_train_cost": best_train,
            "generations": gen,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[seed {seed}] gate {gate_cost:.1f}  learned {learned_mean:.1f} +/- {learned_sem:.1f}  "
              f"savings {savings:+.2f}%  ({rec['seconds']}s, {gen} gens)")
        return rec

    # Shared seed loop + aggregation + >=5-seed enforcement + verdict (single source of truth).
    result = srp.run_over_seeds(PROBLEM_ID, train_one_seed, seeds=seeds)

    out = {
        "reference": parsed.reference,
        "budget": budget_name,
        "smoke": bool(parsed.smoke),
        "structure": {"depth": parsed.depth, "leaf_type": parsed.leaf_type,
                      "split_type": _ar.SPLIT_TYPE, "temperature": parsed.temperature,
                      "action_mode": _ar.ACTION_MODE, "policy_action_mode": _ar.POLICY_ACTION_MODE,
                      "policy_feature_mode": _ar.POLICY_FEATURE_MODE,
                      "input_dim": input_dim, "param_dim": len(x0),
                      "sigma_init": sigma_init, "popsize": budget["popsize"],
                      "max_generations": budget["generations"],
                      "n_train_seeds": len(train_seeds), "n_eval_seeds": len(eval_seeds)},
        "gate_semantics": ("repo reproduction of the published constant node-base-stock "
                           f"benchmark via simulate_base_stock, {len(GATE_SIM_SEEDS)} sim seeds "
                           f"x {gate_replications} replications (the autoresearch keep/discard "
                           "gate); optimizer-seed independent => gate_seed_std == 0"),
        "published_benchmark_cost": published_benchmark,
        "published_ppo_best_cost_CONTEXT_ONLY": published_ppo_best,
        "warm_start_heldout": {"mean": warm_mean, "sem": warm_sem},
        # {"seeds", "per_seed"} + the standardized srp summary keys.
        **result,
    }

    if parsed.smoke:
        out_dir = PACKAGE_ROOT / "outputs" / PROBLEM_ID / "smoke_seed_robust"
        json_path = out_dir / "seed_robust_report_smoke.json"
    else:
        out_dir = PACKAGE_ROOT / "outputs" / PROBLEM_ID
        json_path = out_dir / "seed_robust_report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{parsed.reference}  budget={budget_name}  smoke={parsed.smoke}")
    print(f"GATE seed-mean    {out['gate_seed_mean']:.1f} +/- {out['gate_seed_std']:.1f} "
          f"(published benchmark {published_benchmark:.0f})")
    print(f"LEARNED seed-mean {out['learned_seed_mean']:.1f} +/- {out['learned_seed_std']:.1f}")
    print(f"SAVINGS vs gate   {out['savings_pct_seed_mean']:+.2f}% +/- "
          f"{out['savings_pct_seed_std']:.2f}%  (beating gate {out['frac_seeds_beating_gate']})")
    if published_ppo_best is not None:
        print(f"PPO best context only (cross-protocol, no head-to-head claim): {published_ppo_best:.0f}")
    print(f"VERDICT (vs same-protocol gate): {out['verdict_vs_same_protocol_gate']}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
