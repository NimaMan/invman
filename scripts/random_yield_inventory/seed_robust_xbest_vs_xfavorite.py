"""Paired xbest-vs-xfavorite controlled experiment on the random_yield primary instance.

OBJECTIVE
  Test the training-path-audit hypothesis: ``es_mp.train`` deploys the CMA-ES
  ``xbest`` (``es.result[0]`` = single best individual on the small training-seed
  batch), which OVERFITS on a disjoint held-out CRN block and inflates BOTH the
  held-out cost AND the cross-optimizer-seed std. The proposed lever is to deploy
  ``xfavorite`` instead (``es.result[5]`` = the CMA-ES distribution MEAN /
  ``es.mean`` back-transformed by ``param_scales``), which averages out per-seed
  overfit and should reduce cross-seed std / improve held-out generalization.

  This script captures BOTH endpoints from the SAME CMA-ES run per seed and
  evaluates BOTH on the SAME held-out CRN block (paired, apples-to-apples; the
  only change is which endpoint is deployed). It does NOT modify the global
  default of ``es_mp.train``/``cmaes.py`` -- it extracts xfavorite via the result
  tuple (``CMAES.current_param()``) WITHOUT changing the default.

ALGORITHM (full description)
  1. Build the random_yield d1 oblique-linear soft-tree model (the LOSS-row config:
     800 episodes, train_seed_batch 8, es_population 16, sigma_init 1.5 -- the
     defaults of train_soft_tree_reference.py).
  2. For each optimizer seed s in the >=5-seed block:
       a. Instantiate ``CMAES`` (seed=s) and run a faithful copy of the
          ``es_mp.train`` loop: per episode draw popsize candidates, draw
          per-individual training seeds via ``Seeder`` (same_seed=False, the
          trainer default), score each candidate as the MEAN discounted cost over
          ``train_seed_batch`` consecutive demand seeds (the population-rollout
          Rust kernel), sort ascending by cost, ``es.tell``.
       b. After the last episode capture xbest = ``es.best_param()`` (result[0])
          and xfavorite = ``es.current_param()`` (result[5]).
       c. Evaluate BOTH endpoints on the SAME disjoint held-out CRN block
          (seeds 100000..100000+holdout_seeds-1) and the LIR gate on that block.
  3. Report per-endpoint, per-seed held-out mean; then seed-MEAN +/- cross-seed
     STD, and #seeds beating the LIR gate, for xbest vs xfavorite side by side.

  The loop mirrors ``invman/es_mp.py::train`` exactly (Seeder init_seed=0,
  next_batch_seeds, sort by cost ascending, es.tell on costs) so the xbest path
  reproduces the production-trainer deployment; only the extra xfavorite readout
  is added.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cmaes import CMAES
from invman.utils import Seeder

from common import (
    build_soft_tree_model,
    default_action_cap,
    evaluate_heuristic_policy,
    evaluate_soft_tree_policy,
    get_primary_reference,
    linear_inflation_params,
    soft_tree_rollout_kwargs,
)

import invman_rust


def _population_costs(reference, model, rollout_kwargs, params_batch, seeds, train_seed_batch):
    """Mean discounted cost per candidate over train_seed_batch consecutive seeds."""
    batch_costs = []
    for seed_offset in range(int(train_seed_batch)):
        batch_costs.append(
            invman_rust.random_yield_inventory_soft_tree_population_rollout(
                params_batch=params_batch,
                seeds=[int(seed) + seed_offset for seed in seeds],
                demand_distribution="poisson",
                **rollout_kwargs,
            )
        )
    return np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)


def train_capture_both_endpoints(reference, model, *, seed, episodes, es_population,
                                  sigma_init, train_seed_batch):
    """Faithful copy of es_mp.train's loop; return (xbest_flat, xfavorite_flat)."""
    es = CMAES(model.num_params, sigma_init=sigma_init, popsize=es_population, seed=int(seed))
    seeder = Seeder()  # init_seed=0, matches es_mp.train
    rollout_kwargs = {
        key: value
        for key, value in soft_tree_rollout_kwargs(
            reference, model, flat_params=model.get_model_flat_params()
        ).items()
        if key != "flat_params"
    }
    for _episode in range(1, int(episodes) + 1):
        solutions = es.ask(popsize=es_population)
        seeds = seeder.next_batch_seeds(es_population)  # same_seed=False (trainer default)
        params_batch = [np.asarray(p, dtype=np.float32).tolist() for p in solutions]
        costs = _population_costs(
            reference, model, rollout_kwargs, params_batch, seeds, train_seed_batch
        )
        pop_fitness = [(-float(c), i) for i, c in enumerate(costs.tolist())]
        pop_fitness = sorted(pop_fitness, key=lambda item: item[1])
        es_fitness = np.array([f for f, _ in pop_fitness], dtype=np.float64)
        es.tell(es_fitness)
    xbest = np.asarray(es.best_param(), dtype=np.float32)        # result[0]
    xfavorite = np.asarray(es.current_param(), dtype=np.float32)  # result[5] = es.mean
    return xbest, xfavorite


def main():
    parser = argparse.ArgumentParser(
        description="Paired xbest-vs-xfavorite held-out experiment on random_yield (LOSS-row config)."
    )
    parser.add_argument("--seeds", type=str, default="123,456,789,2026,555")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--leaf_type", type=str, default="linear")
    parser.add_argument("--split_type", type=str, default="oblique")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--training_episodes", type=int, default=800)
    parser.add_argument("--es_population", type=int, default=16)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--train_seed_batch", type=int, default=8)
    parser.add_argument("--holdout_seed_start", type=int, default=100000)
    parser.add_argument("--holdout_seeds", type=int, default=4096)
    parser.add_argument("--output_json", type=str, default=None)
    parsed = parser.parse_args()

    seeds = [int(s) for s in parsed.seeds.split(",") if s.strip()]
    reference = get_primary_reference()
    holdout = [parsed.holdout_seed_start + i for i in range(parsed.holdout_seeds)]

    # LIR gate on the SAME held-out block (deterministic CRN re-eval).
    lir_eval = evaluate_heuristic_policy(reference, "linear_inflation", holdout)
    gate_cost = float(lir_eval["mean_cost"])

    print(f"random_yield xbest-vs-xfavorite | d{parsed.depth} {parsed.leaf_type} "
          f"{parsed.training_episodes}ep pop{parsed.es_population} batch{parsed.train_seed_batch}")
    print(f"held-out CRN block: seeds {parsed.holdout_seed_start}..{parsed.holdout_seed_start+parsed.holdout_seeds-1}")
    print(f"LIR gate (held-out) = {gate_cost:.3f}")
    print(f"{'seed':>6} {'xbest':>12} {'xfavorite':>12} {'gap_xb%':>9} {'gap_xf%':>9}")

    per_seed = []
    for s in seeds:
        model = build_soft_tree_model(
            reference, depth=parsed.depth, temperature=parsed.temperature,
            split_type=parsed.split_type, leaf_type=parsed.leaf_type,
            action_cap=default_action_cap(reference),
        )
        t0 = time.time()
        xbest, xfavorite = train_capture_both_endpoints(
            reference, model, seed=s, episodes=parsed.training_episodes,
            es_population=parsed.es_population, sigma_init=parsed.sigma_init,
            train_seed_batch=parsed.train_seed_batch,
        )
        xb_eval = evaluate_soft_tree_policy(reference, model, holdout, flat_params=xbest.tolist())
        xf_eval = evaluate_soft_tree_policy(reference, model, holdout, flat_params=xfavorite.tolist())
        xb_cost = float(xb_eval["mean_cost"])
        xf_cost = float(xf_eval["mean_cost"])
        rec = {
            "seed": s,
            "xbest_holdout_mean": xb_cost,
            "xfavorite_holdout_mean": xf_cost,
            "xbest_gap_pct_vs_gate": (xb_cost / gate_cost - 1.0) * 100.0,
            "xfavorite_gap_pct_vs_gate": (xf_cost / gate_cost - 1.0) * 100.0,
            "seconds": time.time() - t0,
        }
        per_seed.append(rec)
        print(f"{s:>6} {xb_cost:>12.3f} {xf_cost:>12.3f} "
              f"{rec['xbest_gap_pct_vs_gate']:>+8.2f} {rec['xfavorite_gap_pct_vs_gate']:>+8.2f}")

    xb = [r["xbest_holdout_mean"] for r in per_seed]
    xf = [r["xfavorite_holdout_mean"] for r in per_seed]
    n = len(per_seed)

    def summarize(vals, label):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        below = sum(1 for v in vals if v < gate_cost)
        gap = (mean / gate_cost - 1.0) * 100.0
        print(f"  {label:>10}: {mean:.3f} +/- {std:.3f}  (gap {gap:+.2f}% vs gate {gate_cost:.3f}; "
              f"{below}/{n} below gate)")
        return {"seed_mean": mean, "cross_seed_std": std, "gap_pct_vs_gate": gap,
                "n_below_gate": below, "n_seeds": n}

    print("\nSEED-ROBUST SUMMARY (held-out, paired):")
    xb_summary = summarize(xb, "xbest")
    xf_summary = summarize(xf, "xfavorite")
    std_reduction = (1.0 - xf_summary["cross_seed_std"] / xb_summary["cross_seed_std"]) * 100.0 \
        if xb_summary["cross_seed_std"] > 0 else float("nan")
    print(f"\n  cross-seed std reduction xbest->xfavorite: {std_reduction:+.1f}%")
    print(f"  seed-mean change xbest->xfavorite: {xf_summary['seed_mean'] - xb_summary['seed_mean']:+.3f}")

    payload = {
        "config": {
            "depth": parsed.depth, "leaf_type": parsed.leaf_type, "split_type": parsed.split_type,
            "training_episodes": parsed.training_episodes, "es_population": parsed.es_population,
            "sigma_init": parsed.sigma_init, "train_seed_batch": parsed.train_seed_batch,
            "seeds": seeds,
            "holdout_seed_start": parsed.holdout_seed_start, "holdout_seeds": parsed.holdout_seeds,
        },
        "gate_cost": gate_cost,
        "gate_policy": "linear_inflation",
        "per_seed": per_seed,
        "xbest_summary": xb_summary,
        "xfavorite_summary": xf_summary,
        "cross_seed_std_reduction_pct": std_reduction,
    }
    if parsed.output_json:
        out = Path(parsed.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
