"""Seed-robust learned-policy vs CDI on the Tier-C (hardest demonstrable) dual-sourcing cell.

OBJECTIVE
  The CDI-optimality taxonomy (docs/benchmarks/DUAL_SOURCING_INSTANCE_TAXONOMY_2026_06_07/README.md)
  identifies dual_l2_ce110_b50_u08_catC (l_r=2, c_e=110, b=50, demand U[0,8]) as the hardest
  *reachable* cell: CDI's gap to the bounded-DP optimum is +0.305% single-path / +0.160%
  out-of-sample -- i.e. SMALLER than the path-to-path sampling noise. The expectation is
  therefore a seed-robust MATCH (parity), not a beat. This driver measures that honestly.

ALGORITHM
  1. Resolve the Tier-C reference instance and the CDI-warm-start soft-tree spec
     (soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets), which lives in the
     factorized capped-dual-index coordinate (s_e, Delta_r, cap_r) with CMA-ES warm-started
     at the encoded CDI solution + a best-of{xbest,xfavorite} honest floor.
  2. For each of >=5 optimizer seeds: train (CMA-ES) and evaluate the learned policy on the
     shared CRN eval protocol; read the learned mean cost and the CDI (= best heuristic) mean
     cost on the same protocol.
  3. Report per-seed learned, CDI, gap% vs CDI; then mean +/- std over seeds and the count of
     seeds that strictly beat CDI. Verdict = robust-beat only if mean+std < 0 AND >=4/5 below.

USAGE
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/dual_sourcing/seed_robust_learned_vs_cdi_tier_c.py \
      --seeds 9001 9002 9003 9004 9005 --budget screening --mp_num_processors 2
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dual_sourcing_benchmark_lib as lib
from invman.experiment_runner import run_experiment

INSTANCE = "dual_l2_ce110_b50_u08_catC"
SPEC = next(s for s in lib.EXPERIMENT_SPECS
            if s["id"] == "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[9001, 9002, 9003, 9004, 9005])
    p.add_argument("--budget", default="screening")
    p.add_argument("--mp_num_processors", type=int, default=2)
    p.add_argument("--run_tag", default="ds_catC_seedrobust")
    p.add_argument("--eval_horizon", type=int, default=None)
    p.add_argument("--eval_seeds", type=int, default=None)
    return p.parse_args()


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "benchmarks" / parsed.run_tag
    (root / "results").mkdir(parents=True, exist_ok=True)

    # CDI (best heuristic) on the shared protocol -- seed-independent reference cost.
    ref_args = lib.build_reference_args(INSTANCE)
    heur = lib.evaluate_default_heuristics(ref_args)
    cdi_cost = float(heur["capped_dual_index"]["mean_cost"])
    best_name = min(heur, key=lambda k: heur[k]["mean_cost"])
    best_cost = float(heur[best_name]["mean_cost"])

    rows = []
    for sd in parsed.seeds:
        # The result filename derives from run_tag+reference+spec (NOT the seed), so make
        # the run_tag seed-distinct or every seed would reload the first seed's cached JSON.
        cfg = SimpleNamespace(seed=sd, budget=parsed.budget, run_tag=f"{parsed.run_tag}_s{sd}",
                              mp_num_processors=parsed.mp_num_processors, same_seed=False,
                              eval_horizon=parsed.eval_horizon, eval_seeds=parsed.eval_seeds,
                              training_episodes=None, training_horizon=None)
        args = lib.configure_run_args(cfg, SPEC, root, INSTANCE)
        payload, _ = (json.loads(lib.result_path_for(args).read_text()), None) \
            if lib.result_path_for(args).exists() else run_experiment(args)
        learned = lib.learned_cost_of(payload)
        gap = 100.0 * (learned - cdi_cost) / cdi_cost
        rows.append({"seed": sd, "learned": learned, "gap_pct_vs_cdi": gap})
        print(f"seed {sd}: learned={learned:.4f}  CDI={cdi_cost:.4f}  gap%vsCDI={gap:+.4f}")
        sys.stdout.flush()

    gaps = [r["gap_pct_vs_cdi"] for r in rows]
    learned_costs = [r["learned"] for r in rows]
    mean_g = statistics.mean(gaps)
    std_g = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
    mean_l = statistics.mean(learned_costs)
    std_l = statistics.pstdev(learned_costs) if len(learned_costs) > 1 else 0.0
    n_beat = sum(1 for g in gaps if g < 0)
    robust_beat = (mean_g + std_g < 0) and (n_beat >= max(4, len(gaps) - 1))
    verdict = "robust-beat" if robust_beat else ("robust-match/parity" if abs(mean_g) < 0.5 else "robust-loss")

    print("\n=== Tier-C seed-robust learned vs CDI ===")
    print(f"instance: {INSTANCE}  spec: {SPEC['id']}")
    print(f"best heuristic: {best_name} {best_cost:.4f} (CDI {cdi_cost:.4f})")
    print(f"learned mean+/-std: {mean_l:.4f} +/- {std_l:.4f}  (n={len(rows)} optimizer seeds)")
    print(f"gap%vsCDI mean+/-std: {mean_g:+.4f} +/- {std_g:.4f}   #beat: {n_beat}/{len(rows)}")
    print(f"VERDICT: {verdict}")

    out = {"instance": INSTANCE, "spec": SPEC["id"], "cdi_cost": cdi_cost,
           "best_heuristic_name": best_name, "best_heuristic_cost": best_cost,
           "per_seed": rows, "learned_mean": mean_l, "learned_std": std_l,
           "gap_pct_vs_cdi_mean": mean_g, "gap_pct_vs_cdi_std": std_g,
           "n_beat": n_beat, "n_seeds": len(rows), "verdict": verdict}
    (root / "seed_robust_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root/'seed_robust_summary.json'}")


if __name__ == "__main__":
    main()
