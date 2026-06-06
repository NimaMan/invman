"""Paired xbest-vs-xfavorite seed-robust driver for OWMR het-3 (kaynov2024_instance_12).

OBJECTIVE
  Case (b) of the training-path audit: on the marginal OWMR het-3 instance (the
  partial-backorder K=3 heterogeneous instance where the deployed xbest already
  beats the gate +4.63% +/- 5.51 over 6 seeds), test whether deploying the CMA-ES
  distribution-MEAN endpoint xfavorite (es.current_param() = result[5]) tightens
  the cross-seed std vs the single-best-individual endpoint xbest
  (es.best_param() = result[0]).

  This driver reuses run_asymmetric_learned_vs_gate.run_one VERBATIM with the
  ADDITIVE deploy_endpoint hook. Each run records BOTH endpoints' held-out cost on
  the SAME held-out CRN block (paired), so a single sweep yields the full xbest and
  xfavorite seed blocks. No env change; the global train() default is untouched.

ALGORITHM
  1. For each optimizer seed s in the >=5-seed block, call run_one(... deploy_endpoint
     ='floor', warm_start=True) on kaynov2024_instance_12 with the het-3 geometry
     (echelon_targets_with_alloc_targets, linear leaf, depth-2, axis_aligned,
     absolute_augmented state, train_allocation min_shortage, same_seed). The gate
     grid search is cached across seeds (same instance/budget/alloc set).
  2. From each run's result read xbest_cost and xfavorite_cost (held-out means on
     the SAME paired block) and the gate_cost.
  3. Report per-endpoint seed-MEAN +/- cross-seed STD, gap% vs gate, and #seeds
     beating the gate -- xbest vs xfavorite side by side -- plus the std reduction.

  Budget is reduced from the full 600/pop32/batch24/holdout4096 to keep 5 seeds
  inside the CPU/time cap (see --training_episodes etc.); the reduction is recorded
  in the output JSON. The xbest seed block from this driver is the apples-to-apples
  comparator for xfavorite (same runs), NOT a re-derivation of the prior 6-seed
  +4.63% headline (which used the full budget).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_asymmetric_learned_vs_gate import run_one  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="OWMR het-3 paired xbest-vs-xfavorite seed sweep.")
    p.add_argument("--reference", default="kaynov2024_instance_12")
    p.add_argument("--seeds", default="821,822,823,824,825")
    p.add_argument("--budget", default="full")
    p.add_argument("--leaf_type", default="linear")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--split_type", default="axis_aligned")
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--policy_action_mode", default="echelon_targets_with_alloc_targets")
    p.add_argument("--policy_state_mode", default="absolute_augmented")
    p.add_argument("--train_allocation", default="min_shortage")
    p.add_argument("--sigma_init", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--training_episodes", type=int, default=300)
    p.add_argument("--es_population", type=int, default=24)
    p.add_argument("--train_seed_batch", type=int, default=12)
    p.add_argument("--holdout_paths", type=int, default=2048)
    p.add_argument("--gate_search_paths", type=int, default=128)
    p.add_argument("--output_json", default=None)
    parsed = p.parse_args()

    import run_asymmetric_learned_vs_gate as runner
    reference = runner.common.get_reference(parsed.reference)
    seeds = [int(s) for s in parsed.seeds.split(",") if s.strip()]
    out_root = PACKAGE_ROOT / "outputs" / "one_warehouse_multi_retailer" / "asymmetric_learned"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"OWMR het-3 xbest-vs-xfavorite | {parsed.reference} {parsed.policy_action_mode} "
          f"d{parsed.depth} {parsed.leaf_type} gen{parsed.training_episodes} pop{parsed.es_population} "
          f"batch{parsed.train_seed_batch} holdout{parsed.holdout_paths}")
    print(f"{'seed':>6} {'gate':>10} {'xbest':>10} {'xfavorite':>10} {'gap_xb%':>9} {'gap_xf%':>9} {'deployed':>20}")

    per_seed = []
    for s in seeds:
        t0 = time.time()
        res = run_one(
            reference, parsed.budget, parsed.leaf_type, parsed.policy_action_mode,
            parsed.train_allocation, s, parsed.sigma_init, True, parsed.workers, out_root,
            gate_search_paths=parsed.gate_search_paths,
            depth=parsed.depth, temperature=parsed.temperature, split_type=parsed.split_type,
            training_episodes=parsed.training_episodes, es_population=parsed.es_population,
            train_seed_batch=parsed.train_seed_batch, holdout_paths=parsed.holdout_paths,
            policy_state_mode=parsed.policy_state_mode, same_seed=True,
            deploy_endpoint="floor",
        )
        rec = {
            "seed": s,
            "gate_cost": res["gate_cost"],
            "xbest_cost": res["xbest_cost"],
            "xfavorite_cost": res["xfavorite_cost"],
            "xbest_gap_pct_vs_gate": res["xbest_gap_pct_vs_gate"],
            "xfavorite_gap_pct_vs_gate": res["xfavorite_gap_pct_vs_gate"],
            "deployed_policy": res["deployed_policy"],
            "seconds": time.time() - t0,
        }
        per_seed.append(rec)
        print(f"{s:>6} {rec['gate_cost']:>10.2f} {rec['xbest_cost']:>10.2f} {rec['xfavorite_cost']:>10.2f} "
              f"{rec['xbest_gap_pct_vs_gate']:>+8.2f} {rec['xfavorite_gap_pct_vs_gate']:>+8.2f} {rec['deployed_policy']:>20}")

    n = len(per_seed)
    gate_mean = statistics.mean(r["gate_cost"] for r in per_seed)

    def summarize(key, label):
        vals = [r[key] for r in per_seed]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        # "beat gate" = lower cost than that seed's gate (paired)
        below = sum(1 for r in per_seed if r[key] < r["gate_cost"])
        gap = (gate_mean - mean) / gate_mean * 100.0  # +ve => learned cheaper (savings)
        print(f"  {label:>10}: {mean:.3f} +/- {std:.3f}  (savings {gap:+.2f}% vs gate {gate_mean:.2f}; "
              f"{below}/{n} below gate)")
        return {"seed_mean": mean, "cross_seed_std": std, "savings_pct_vs_gate": gap,
                "n_below_gate": below, "n_seeds": n}

    print("\nSEED-ROBUST SUMMARY (held-out, paired):")
    xb = summarize("xbest_cost", "xbest")
    xf = summarize("xfavorite_cost", "xfavorite")
    std_reduction = (1.0 - xf["cross_seed_std"] / xb["cross_seed_std"]) * 100.0 \
        if xb["cross_seed_std"] > 0 else float("nan")
    print(f"\n  cross-seed std reduction xbest->xfavorite: {std_reduction:+.1f}%")
    print(f"  seed-mean change xbest->xfavorite: {xf['seed_mean'] - xb['seed_mean']:+.3f}")

    payload = {
        "config": {
            "reference": parsed.reference, "policy_action_mode": parsed.policy_action_mode,
            "policy_state_mode": parsed.policy_state_mode, "leaf_type": parsed.leaf_type,
            "depth": parsed.depth, "split_type": parsed.split_type, "temperature": parsed.temperature,
            "train_allocation": parsed.train_allocation, "sigma_init": parsed.sigma_init,
            "training_episodes": parsed.training_episodes, "es_population": parsed.es_population,
            "train_seed_batch": parsed.train_seed_batch, "holdout_paths": parsed.holdout_paths,
            "gate_search_paths": parsed.gate_search_paths, "seeds": seeds,
            "budget_reduced_from_full": True,
        },
        "gate_mean": gate_mean,
        "per_seed": per_seed,
        "xbest_summary": xb,
        "xfavorite_summary": xf,
        "cross_seed_std_reduction_pct": std_reduction,
    }
    if parsed.output_json:
        out = Path(parsed.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
