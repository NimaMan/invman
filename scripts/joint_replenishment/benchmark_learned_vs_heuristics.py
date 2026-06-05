"""Train and benchmark a learned soft-tree policy on the Vanvuchelen et al. (2020) JRP.

OBJECTIVE
---------
Fill in the "learned needs a run" placeholder left by
scripts/joint_replenishment/benchmark_vanvuchelen_settings.py. For each of the 16
Vanvuchelen, Gijsbrechts & Boute (2020) Table-2 settings, train a soft-tree policy
with CMA-ES (Rust-backed rollouts) and benchmark it against the two carried repo
heuristics (DYN-OUT / MOQ). The Figure-3 optimal action map for setting 5
(joint_replenishment_published_action_anchor) is reported as the literature anchor.

ALGORITHMIC DESCRIPTION
-----------------------
For each setting s:
  1. Reference -> rollout config. The 16 Table-2 references carry only the cost /
     demand structure. We add the missing simulation knobs used everywhere else in
     this problem's scripts: horizon = `periods` (default 200), initial inventory
     = zeros, discount gamma = 0.99, and a generous action box per item of
     2 * truck_capacity (>= any sensible base-stock order; keeps the soft-tree
     vector_quantity action map from being starved).
  2. Soft-tree policy. backbone='soft_tree', input_dim = num_items + 2 (the env
     feature vector: per-item inventory levels + total-inventory + remaining-horizon
     fraction), control_mode='vector_quantity', control_dim = num_items.
  3. CMA-ES training (invman.es_mp.train, population path). Each generation samples
     `es_population` candidate parameter vectors; the Rust population rollout scores
     each candidate on its own paired seed (common random numbers within the
     generation), averaged over `train_seed_batch` seed offsets, and returns the
     negative discounted cost as fitness. Seeds advance each generation.
  4. HELD-OUT EVALUATION (the anti-overfitting protocol). Training seeds are drawn
     from a base block (default seed .. seed + train horizon). Evaluation uses a
     DISJOINT block of `eval_seeds` seeds starting at EVAL_SEED_BASE (default
     1_000_000), well past any seed CMA-ES could have visited. The SAME eval-seed
     block is reused (paired / CRN) for the learned policy and both heuristics, so
     learned-vs-heuristic gaps are variance-reduced common-random-number comparisons.
  5. Reporting. Per setting: learned mean discounted cost, MOQ mean, DYN-OUT mean,
     best heuristic, gap, %win, and which policy wins. Setting 5 additionally lists
     the published Figure-3 optimal/heuristic action anchor.

CPU CAP
-------
This script is written to run UNDER A HARD 2-CORE CAP (other CMA-ES agents run in
parallel). The Rust population rollout uses rayon; the shared CPU helper caps native
threads at import time (before invman_rust is loaded), and mp_num_processors=1 ensures the
es_mp Pool fallback (unused on the population path) cannot fan out either. Lower externally
exported thread values are preserved; higher values are capped by the shared CPU helper.

USAGE
-----
    RAYON_NUM_THREADS=2 python benchmark_learned_vs_heuristics.py
    RAYON_NUM_THREADS=2 python benchmark_learned_vs_heuristics.py --settings vanvuchelen2020_small_scale_setting_5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

# HARD 2-CORE CAP. Other CMA-ES agents run in parallel, so every layer that can
# spin up threads must be capped BEFORE numpy / invman_rust import their native libs.
configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

from invman.es_mp import train

from common import (
    build_soft_tree_model,
    dumps_json,
    ensure_parent,
    evaluate_heuristic_policy,
    evaluate_soft_tree_policy,
    list_references,
    newsvendor_item_targets,
    soft_tree_rollout_kwargs,
)

import invman_rust


EVAL_SEED_BASE = 1_000_000  # disjoint from any training seed block


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="*", default=None,
                        help="subset of setting names; default = all 16 Table-2 settings")
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--training_episodes", type=int, default=120)
    parser.add_argument("--es_population", type=int, default=24)
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_seed_batch", type=int, default=4)
    parser.add_argument("--eval_seeds", type=int, default=2048)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _full_reference(ref: dict, parsed) -> dict:
    """Attach the simulation knobs the 16 Table-2 references omit."""
    full = dict(ref)
    full["periods"] = int(parsed.periods)
    full["discount_factor"] = float(parsed.discount_factor)
    full["initial_inventory_levels"] = [0] * int(ref.get("num_items", len(ref["demand_highs"])))
    return full


def _training_namespace(parsed, reference):
    run_tag = (
        f"jrp_{reference['name']}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_s{parsed.seed}_b{parsed.train_seed_batch}"
    )
    output_root = PACKAGE_ROOT / "outputs" / "joint_replenishment" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,  # population path bypasses the Pool; keep the fallback single-process too
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(reference["periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_model_fitness(model, reference):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1,
              return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            costs.append(float(invman_rust.joint_replenishment_soft_tree_rollout(
                seed=int(seed) + seed_offset,
                **soft_tree_rollout_kwargs(reference, model, flat_params=flat_params),
            )))
        reward = -float(np.mean(costs))
        if verbose:
            print(f"Seed {seed}: reward {reward:.4f}")
        return reward, indiv_idx

    return inner


def _get_population_fitness(model, reference):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(p, dtype=np.float32).tolist() for p in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference, model, flat_params=model.get_model_flat_params()
            ).items()
            if key != "flat_params"
        }
        batch_costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            batch_costs.append(
                invman_rust.joint_replenishment_soft_tree_population_rollout(
                    params_batch=params_batch,
                    seeds=[int(s) + seed_offset for s in seeds],
                    **rollout_kwargs,
                )
            )
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [(-float(c), i) for i, c in enumerate(costs.tolist())]

    return inner


def _train_and_evaluate(ref: dict, parsed) -> dict:
    reference = _full_reference(ref, parsed)
    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
    )
    train_args = _training_namespace(parsed, reference)
    trained_model, _ = train(
        model=model,
        get_model_fitness=_get_model_fitness(model, reference),
        get_population_fitness=_get_population_fitness(model, reference),
        args=train_args,
        same_seed=False,
    )

    eval_seeds = [EVAL_SEED_BASE + offset for offset in range(int(parsed.eval_seeds))]
    learned = evaluate_soft_tree_policy(reference, trained_model, eval_seeds)

    targets = newsvendor_item_targets(reference)
    moq = evaluate_heuristic_policy(
        reference, "minimum_order_quantity",
        [float(targets[0]), float(targets[1]), 1.0, 2.0],
        replications=int(parsed.eval_seeds), seed=EVAL_SEED_BASE,
    )
    dynout = evaluate_heuristic_policy(
        reference, "dynamic_order_up_to",
        [float(targets[0]), float(targets[1])],
        replications=int(parsed.eval_seeds), seed=EVAL_SEED_BASE,
    )

    learned_cost = float(learned["mean_cost"])
    heuristics = {"DYN-OUT": float(dynout["mean_cost"]), "MOQ": float(moq["mean_cost"])}
    best_heur_name = min(heuristics, key=heuristics.get)
    best_heur_cost = heuristics[best_heur_name]
    gap = best_heur_cost - learned_cost  # positive => learned cheaper than best heuristic
    pct_win = 100.0 * gap / best_heur_cost if best_heur_cost else 0.0
    winner = "learned" if learned_cost < best_heur_cost - 1e-9 else best_heur_name

    return {
        "name": ref["name"],
        "newsvendor_targets": [int(t) for t in targets],
        "learned_mean_cost": learned_cost,
        "learned_cost_std": float(learned["cost_std"]),
        "moq_mean_cost": float(moq["mean_cost"]),
        "dynout_mean_cost": float(dynout["mean_cost"]),
        "best_heuristic": best_heur_name,
        "best_heuristic_cost": best_heur_cost,
        "gap_best_heuristic_minus_learned": gap,
        "pct_win_vs_best_heuristic": pct_win,
        "winner": winner,
        "num_eval_seeds": int(parsed.eval_seeds),
    }


def _markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Setting | Learned | DYN-OUT | MOQ | Best heuristic | Gap (best-learned) | %win | Winner |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | `{r['learned_mean_cost']:.2f}` | `{r['dynout_mean_cost']:.2f}` | "
            f"`{r['moq_mean_cost']:.2f}` | {r['best_heuristic']} | "
            f"`{r['gap_best_heuristic_minus_learned']:+.2f}` | `{r['pct_win_vs_best_heuristic']:+.2f}%` | "
            f"**{r['winner']}** |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    refs = list_references()
    if parsed.settings:
        wanted = set(parsed.settings)
        refs = [r for r in refs if r["name"] in wanted]
        missing = wanted - {r["name"] for r in refs}
        if missing:
            raise SystemExit(f"unknown settings: {sorted(missing)}")

    anchor = dict(invman_rust.joint_replenishment_published_action_anchor())

    print("=" * 90)
    print("LEARNED soft-tree vs repo heuristics (DYN-OUT / MOQ) -- Vanvuchelen (2020) JRP")
    print(f"  budget: depth={parsed.depth} pop={parsed.es_population} gens={parsed.training_episodes} "
          f"sigma0={parsed.sigma_init} train_seed_batch={parsed.train_seed_batch}")
    print(f"  eval: {parsed.eval_seeds} held-out CRN seeds from base {EVAL_SEED_BASE} "
          f"(disjoint from train seed {parsed.seed}); horizon={parsed.periods}, gamma={parsed.discount_factor}")
    print(f"  RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}  mp_num_processors=1")
    print("=" * 90)

    rows = []
    for ref in refs:
        row = _train_and_evaluate(ref, parsed)
        rows.append(row)
        print(f"  {row['name']:<42} learned={row['learned_mean_cost']:>9.2f} "
              f"DYN-OUT={row['dynout_mean_cost']:>9.2f} MOQ={row['moq_mean_cost']:>9.2f} "
              f"-> {row['winner']} ({row['pct_win_vs_best_heuristic']:+.2f}% vs {row['best_heuristic']})")

    md = _markdown_table(rows)
    print()
    print(md)

    payload = {
        "budget": {
            "depth": parsed.depth,
            "temperature": parsed.temperature,
            "split_type": parsed.split_type,
            "leaf_type": parsed.leaf_type,
            "training_episodes": parsed.training_episodes,
            "es_population": parsed.es_population,
            "sigma_init": parsed.sigma_init,
            "seed": parsed.seed,
            "train_seed_batch": parsed.train_seed_batch,
            "eval_seeds": parsed.eval_seeds,
            "eval_seed_base": EVAL_SEED_BASE,
            "periods": parsed.periods,
            "discount_factor": parsed.discount_factor,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "mp_num_processors": 1,
        },
        "published_action_anchor": anchor,
        "rows": rows,
        "markdown": md,
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {output_path}")

    print()
    print(dumps_json({"published_action_anchor": anchor}))


if __name__ == "__main__":
    main()
