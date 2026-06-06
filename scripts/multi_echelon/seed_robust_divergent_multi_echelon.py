"""
Seed-ROBUST learned-vs-gate runner for the multi_echelon DIVERGENT special-delivery
family (Gijsbrechts 2022 setting1 & setting2) -- the best-of-N "beats A3C 14.4%" fix.

OBJECTIVE
---------
The at-risk paper claim is: "divergent setting1: learned 779.81 vs best constant base-stock
911.39 -> -14.44%, exceeding A3C savings 8.95%" (and setting2 -14.43% vs A3C 12.09%), with
`seed_reporting: best_of_n`. Per the project mandate, a "beats X" headline must be a
MEAN +/- STD over >=5 INDEPENDENT optimizer seeds, never a single seed / best-of-N. The A3C
8.95% / 12.09% numbers are CROSS-PROTOCOL CONTEXT (the repo implements no A3C), so the only
SAME-PROTOCOL comparator is the env's own in-region best constant base-stock gate; we report
learned savings vs that gate as mean +/- std over >=5 seeds.

ALGORITHM (full description)
----------------------------
1. For each optimizer seed: call the EXISTING train_multi_echelon_policy helpers (no new env):
   - best_constant_base_stock_over_operating_region(...) -> the SAME-protocol gate at this seed
     (the gate grid search is itself seeded, so it moves slightly per seed; we record it).
   - train_one(...) for each (design, depth) in the sweep -> learned held-out mean cost.
   - learned savings % = 100 * (gate - best_learned) / gate, with the gate AT THE SAME SEED
     (paired; this is the honest same-protocol margin).
2. Aggregate across seeds: report the learned-cost seed-mean +/- std, the gate seed-mean +/-
   std, and the SAVINGS-% seed-mean +/- std (the headline). Verdict:
     ROBUST_BEAT_VS_GATE  if savings_mean > savings_std and every seed's savings > 0;
     PARITY               if |savings_mean| <= savings_std;
     ROBUST_LOSS          if savings_mean < -savings_std.
   The published A3C savings is printed ALONGSIDE as cross-protocol context only -- never a
   head-to-head verdict.

CPU CAP / USAGE
---------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/multi_echelon/seed_robust_divergent_multi_echelon.py \
      --reference gijsbrechts2022_setting1 --budget full --designs direct_level \
      --depths 2 3 --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv  # noqa: E402

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import invman_rust  # noqa: F401,E402

# Load the existing single-seed runner as a module and reuse its functions verbatim.
_RUNNER = PACKAGE_ROOT / "scripts" / "multi_echelon" / "train_multi_echelon_policy.py"
_spec = importlib.util.spec_from_file_location("train_multi_echelon_policy", _RUNNER)
_tmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tmp)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="gijsbrechts2022_setting1")
    p.add_argument("--budget", choices=sorted(_tmp.BUDGETS), default="full")
    p.add_argument("--designs", default="direct_level")
    p.add_argument("--depths", default="2,3")
    p.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002, 9003, 9004, 9005])
    p.add_argument("--mp_num_processors", type=int, default=2)
    p.add_argument("--sigma_init", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--run_tag", default="divergent_seed_robust")
    # ADDITIVE/REVERSIBLE (training-path audit 2026-06-06): which CMA-ES endpoint(s)
    # train_one may deploy. "floor" (default) deploys the cheaper of {xbest,
    # xfavorite} per (design, depth) on the SAME eval seeds (downside-safe; never
    # worse than xbest). "xbest" reproduces the historical run_experiment deployment
    # EXACTLY (deploy the single best CMA-ES individual).
    p.add_argument("--deploy_endpoint", choices=["floor", "xbest"], default="floor")
    return p.parse_args()


def main():
    parsed = parse_args()
    budget = _tmp.BUDGETS[parsed.budget]
    reference_name = parsed.reference
    designs = [d.strip() for d in str(parsed.designs).split(",") if d.strip()]
    depths = [int(d) for d in str(parsed.depths).split(",") if d.strip()]
    out_dir = PACKAGE_ROOT / "outputs" / "multi_echelon" / f"{reference_name}_seed_robust"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = _tmp._are.build_reference_args(reference_name)
    a3c = _tmp.published_a3c_savings(reference_name)

    per_seed = []
    for seed in parsed.seeds:
        # A throwaway namespace mimicking the runner's `parsed` for train_one / gate.
        ns = argparse.Namespace(
            reference=reference_name, budget=parsed.budget, seed=seed,
            mp_num_processors=parsed.mp_num_processors, sigma_init=parsed.sigma_init,
            temperature=parsed.temperature,
        )
        t0 = time.time()
        gate = _tmp.best_constant_base_stock_over_operating_region(base_args, budget, seed)
        gate_cost = float(gate["mean_cost"])
        runs = []
        for design in designs:
            for depth in depths:
                run = _tmp.train_one(
                    reference_name, design, depth, budget, ns, out_dir,
                    deploy_endpoint=parsed.deploy_endpoint,
                )
                runs.append(run)
        best = min(runs, key=lambda r: r["mean_cost"])
        savings = 100.0 * (gate_cost - best["mean_cost"]) / gate_cost
        # floor bookkeeping: did the floor ever deploy xfavorite instead of xbest?
        floor_deviated = any(r.get("deployed_endpoint") == "xfavorite" for r in runs)
        per_seed.append({
            "seed": seed,
            "gate_cost": gate_cost,
            "gate_yw": gate["warehouse_level"], "gate_yr": gate["retailer_level"],
            "best_learned_cost": float(best["mean_cost"]),
            "best_design": best["design"], "best_depth": best["depth"],
            "best_deployed_endpoint": best.get("deployed_endpoint"),
            "deploy_endpoint": parsed.deploy_endpoint,
            "floor_deviated_from_xbest": floor_deviated,
            "savings_pct_vs_gate": savings,
            "all_runs": [{"design": r["design"], "depth": r["depth"],
                          "mean_cost": r["mean_cost"], "std_cost": r["std_cost"],
                          "deployed_endpoint": r.get("deployed_endpoint"),
                          "xbest_cost": r.get("xbest_cost"),
                          "xfavorite_cost": r.get("xfavorite_cost")} for r in runs],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[seed {seed}] gate {gate_cost:.2f} (yw={gate['warehouse_level']},yr={gate['retailer_level']})  "
              f"best learned {best['mean_cost']:.2f} ({best['best_design'] if False else best['design']} d{best['depth']})  "
              f"savings {savings:+.2f}%  ({per_seed[-1]['seconds']}s)")

    learned = [s["best_learned_cost"] for s in per_seed]
    gates = [s["gate_cost"] for s in per_seed]
    sav = [s["savings_pct_vs_gate"] for s in per_seed]
    n = len(per_seed)
    sav_mean = statistics.mean(sav)
    sav_std = statistics.stdev(sav) if n > 1 else 0.0
    frac_pos = sum(1 for v in sav if v > 0)

    if sav_mean > sav_std and frac_pos == n and sav_std >= 0:
        verdict = "ROBUST_BEAT_VS_GATE"
    elif abs(sav_mean) <= max(sav_std, 1e-9):
        verdict = "PARITY"
    elif sav_mean < 0:
        verdict = "ROBUST_LOSS_VS_GATE"
    else:
        verdict = "BEAT_WITHIN_STD"

    out = {
        "reference": reference_name,
        "budget": parsed.budget,
        "designs": designs, "depths": depths,
        "n_seeds": n, "seeds": parsed.seeds,
        "deploy_endpoint": parsed.deploy_endpoint,
        "floor_deviated_any_seed": any(s.get("floor_deviated_from_xbest") for s in per_seed),
        "per_seed": per_seed,
        "learned_seed_mean": statistics.mean(learned),
        "learned_seed_std": statistics.stdev(learned) if n > 1 else 0.0,
        "gate_seed_mean": statistics.mean(gates),
        "gate_seed_std": statistics.stdev(gates) if n > 1 else 0.0,
        "savings_pct_seed_mean": sav_mean,
        "savings_pct_seed_std": sav_std,
        "frac_seeds_beating_gate": f"{frac_pos}/{n}",
        "published_a3c_savings_pct_CONTEXT_ONLY": a3c,
        "verdict_vs_same_protocol_gate": verdict,
    }
    json_path = out_dir / f"seed_robust_{parsed.budget}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{reference_name}  budget={parsed.budget}  designs={designs} depths={depths}")
    print(f"GATE seed-mean   {out['gate_seed_mean']:.2f} +/- {out['gate_seed_std']:.2f}")
    print(f"LEARNED seed-mean {out['learned_seed_mean']:.2f} +/- {out['learned_seed_std']:.2f}")
    print(f"SAVINGS vs gate  {sav_mean:+.2f}% +/- {sav_std:.2f}%  (beating gate {frac_pos}/{n})")
    print(f"A3C context only: {a3c}%")
    print(f"VERDICT (vs same-protocol gate): {verdict}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
