"""Learned soft-tree vs heuristic vs published benchmark for one_warehouse_multi_retailer.

OBJECTIVE
---------
Fill the "learned needs a run" placeholders for `one_warehouse_multi_retailer`
(Kaynov et al. 2024, IJPE 267, 109088) by training a Rust-backed soft-tree
policy with CMA-ES and comparing it, OUT OF SAMPLE, against:
  - the repo's best echelon base-stock + allocation heuristic (proportional and
    min_shortage rationing, grid-searched on a search-seed block);
  - the published Kaynov Table A.3 rows (proportional / min_shortage / PPO);
  - the repo-native exact finite-horizon DP self-consistency anchor.

This mirrors the finished problems (perishable_inventory, spare_parts_inventory):
one faithful env, a learned policy trained with `invman.cmaes` via the problem's
rollout binding and `invman.policy.Policy`, evaluated on a DISJOINT held-out seed
block under Common Random Numbers (CRN) against the heuristic.

ALGORITHM (per instance)
------------------------
1. Resolve the Kaynov reference; build the mean-filled warm-start initial state
   (the repo evaluation convention; same as run_heuristic_published_benchmark.py).
2. CRN held-out block: draw `--holdout_paths` demand-path matrices
   (periods x K, the instance's demand distribution) from a numpy RNG seeded at
   `--holdout_seed_start`. This block is DISJOINT from training (training uses
   the es_mp Seeder's own random per-generation seeds and the in-Rust Poisson
   sampler; evaluation supplies explicit precomputed paths via the *_from_paths
   bindings, so there is no seed overlap).
3. Heuristic: grid-search echelon base-stock (W, R) for {proportional,
   min_shortage} on a SEPARATE `--search_paths` block (search-seed block), then
   re-evaluate the argmin on the held-out block. Both allocation rules reported;
   the better one is "best heuristic".
4. Learned: train a soft-tree (`symmetric_echelon_targets` action mode for the
   symmetric K=3 instances) with CMA-ES (`invman.es_mp.train` + the population
   rollout binding), then evaluate the trained weights on the SAME held-out
   paths via the soft-tree *_from_paths binding (CRN-paired with the heuristic).
5. Published comparison: cost = -published_reward (paper reports negative reward).
   Report learned vs best-heuristic vs published-{prop,min,PPO}, the win, and gaps.

PROTOCOL
--------
- 100-period undiscounted total cost (discount_factor = 1.0), matching the paper.
- Training-allocation / eval-allocation split follows the repo convention
  (train with `--train_allocation`, evaluate the learned policy with the SAME
  allocation the heuristic argmin used so the rationing rule is held fixed in the
  comparison; both reported).
- HARD CPU CAP: set RAYON_NUM_THREADS=2 / OMP_NUM_THREADS=2 in the environment
  before launching this script. The CMA-ES path here uses the population rollout
  binding (no Python process pool), so all parallelism is rayon inside Rust and is
  bounded by RAYON_NUM_THREADS. mp_num_processors is pinned to 1 regardless.

USAGE
-----
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/one_warehouse_multi_retailer/benchmark_learned_vs_heuristic.py \
      --instance_names kaynov2024_instance_1 kaynov2024_instance_6 kaynov2024_instance_11 \
      --training_episodes 400 --es_population 24 --train_seed_batch 8 \
      --search_paths 256 --holdout_paths 4096 \
      --output_json outputs/one_warehouse_multi_retailer/learned_benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.es_mp import train  # noqa: E402

from common import (  # noqa: E402
    benchmark_initial_state,
    build_soft_tree_model,
    echelon_base_stock_search_bounds,
    is_symmetric_retailer_case,
    policy_action_mode_for_reference,
    soft_tree_rollout_kwargs,
)

import invman_rust as ir  # noqa: E402

DISCOUNT_FACTOR = 1.0  # paper protocol: 100-period undiscounted total cost


# --------------------------------------------------------------------------- #
# CRN demand paths                                                             #
# --------------------------------------------------------------------------- #
def _sample_demand_paths(reference: dict, num_paths: int, seed: int) -> list[list[list[int]]]:
    """Draw `num_paths` demand-path matrices (periods x K) with the instance's
    per-retailer demand distribution. A single numpy RNG seeded at `seed` drives
    all paths so the block is fully reproducible and disjoint between search and
    held-out (different seeds)."""
    periods = int(reference["benchmark_periods"])
    rng = np.random.RandomState(int(seed))
    kinds = list(reference["demand_kinds"])
    p1 = [float(v) for v in reference["demand_param1"]]
    p2 = [float(v) for v in reference["demand_param2"]]

    columns: list[np.ndarray] = []
    for kind, a, b in zip(kinds, p1, p2):
        if kind == "poisson":
            col = rng.poisson(a, size=(num_paths, periods))
        elif kind == "discrete_uniform":
            col = rng.randint(int(round(a)), int(round(b)) + 1, size=(num_paths, periods))
        elif kind == "deterministic":
            col = np.full((num_paths, periods), int(round(a)), dtype=np.int64)
        elif kind == "rounded_normal":
            raw = rng.normal(a, b, size=(num_paths, periods))
            col = np.clip(np.rint(raw), 0, None).astype(np.int64)
        else:
            raise ValueError(f"unsupported demand kind '{kind}'")
        columns.append(np.asarray(col, dtype=np.int64))

    paths: list[list[list[int]]] = []
    for path_idx in range(num_paths):
        matrix = [[int(columns[k][path_idx, t]) for k in range(len(columns))] for t in range(periods)]
        paths.append(matrix)
    return paths


# --------------------------------------------------------------------------- #
# Heuristic on explicit paths (CRN)                                            #
# --------------------------------------------------------------------------- #
def _heuristic_on_paths(
    reference: dict,
    warehouse_level: int,
    retailer_levels: list[int],
    allocation: str,
    paths: list[list[list[int]]],
    alloc_seed: int,
) -> np.ndarray:
    ist = benchmark_initial_state(reference)
    costs = []
    for path_idx, demands in enumerate(paths):
        cost = ir.one_warehouse_multi_retailer_policy_rollout_from_paths(
            policy_name="echelon_base_stock",
            params=[float(warehouse_level)] + [float(v) for v in retailer_levels],
            initial_warehouse_inventory=int(ist["initial_warehouse_inventory"]),
            initial_warehouse_pipeline=ist["initial_warehouse_pipeline"],
            initial_retailer_inventory=ist["initial_retailer_inventory"],
            initial_retailer_pipeline=ist["initial_retailer_pipeline"],
            demands=demands,
            holding_cost_warehouse=float(reference["holding_cost_warehouse"]),
            holding_cost_retailers=[float(v) for v in reference["holding_cost_retailers"]],
            penalty_costs_retailers=[float(v) for v in reference["penalty_costs_retailers"]],
            customer_behavior=str(reference["customer_behavior"]),
            seed=int(alloc_seed) + path_idx,
            emergency_shipment_probability=float(reference["emergency_shipment_probability"]),
            discount_factor=DISCOUNT_FACTOR,
            allocation_policy=str(allocation),
        )
        costs.append(float(cost))
    return np.asarray(costs, dtype=np.float64)


def _search_best_heuristic_on_paths(
    reference: dict,
    allocation: str,
    search_paths: list[list[list[int]]],
    alloc_seed: int,
) -> dict:
    bounds = echelon_base_stock_search_bounds(reference)
    wlo, whi = bounds["warehouse"]
    symmetric = bounds["symmetric_retailers"]
    K = len(reference["retailer_lead_times"])
    best = None
    if symmetric:
        rlo, rhi = bounds["retailers"][0]
        candidates = (
            (w, [r] * K) for w in range(wlo, whi + 1) for r in range(rlo, rhi + 1)
        )
    else:
        from itertools import product

        grids = [range(lo, hi + 1) for lo, hi in bounds["retailers"]]
        candidates = (
            (w, list(levels)) for w in range(wlo, whi + 1) for levels in product(*grids)
        )
    for w, r in candidates:
        costs = _heuristic_on_paths(reference, w, r, allocation, search_paths, alloc_seed)
        mean_cost = float(costs.mean())
        if best is None or mean_cost < best["mean_cost"]:
            best = {
                "warehouse_base_stock_level": int(w),
                "retailer_base_stock_levels": [int(x) for x in r],
                "mean_cost": mean_cost,
            }
    best["search_bounds"] = bounds
    return best


# --------------------------------------------------------------------------- #
# Soft-tree on explicit paths (CRN)                                            #
# --------------------------------------------------------------------------- #
def _soft_tree_on_paths(
    reference: dict,
    model,
    flat_params,
    allocation: str,
    policy_action_mode: str,
    paths: list[list[list[int]]],
    alloc_seed: int,
) -> np.ndarray:
    ist = benchmark_initial_state(reference)
    flat_list = np.asarray(flat_params, dtype=np.float32).tolist()
    costs = []
    for path_idx, demands in enumerate(paths):
        cost = ir.one_warehouse_multi_retailer_soft_tree_rollout_from_paths(
            flat_params=flat_list,
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            min_values=[int(v) for v in model.min_values],
            max_values=[int(v) for v in model.max_values],
            action_mode=str(model.control_mode),
            initial_warehouse_inventory=int(ist["initial_warehouse_inventory"]),
            initial_warehouse_pipeline=ist["initial_warehouse_pipeline"],
            initial_retailer_inventory=ist["initial_retailer_inventory"],
            initial_retailer_pipeline=ist["initial_retailer_pipeline"],
            demands=demands,
            holding_cost_warehouse=float(reference["holding_cost_warehouse"]),
            holding_cost_retailers=[float(v) for v in reference["holding_cost_retailers"]],
            penalty_costs_retailers=[float(v) for v in reference["penalty_costs_retailers"]],
            customer_behavior=str(reference["customer_behavior"]),
            seed=int(alloc_seed) + path_idx,
            emergency_shipment_probability=float(reference["emergency_shipment_probability"]),
            discount_factor=DISCOUNT_FACTOR,
            allocation_policy=str(allocation),
            policy_action_mode=str(policy_action_mode),
            retailer_target_inventory_positions=None,
            temperature=float(model.temperature),
            split_type=str(model.split_type),
            leaf_type=str(model.leaf_type),
            allowed_values=model.allowed_values,
        )
        costs.append(float(cost))
    return np.asarray(costs, dtype=np.float64)


# --------------------------------------------------------------------------- #
# CMA-ES training (population rollout binding; no Python pool)                 #
# --------------------------------------------------------------------------- #
def _training_namespace(parsed, reference, model) -> SimpleNamespace:
    run_tag = (
        f"learned_{reference['name']}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
        f"_pop{parsed.es_population}_e{parsed.training_episodes}_s{parsed.seed}"
    )
    output_root = PACKAGE_ROOT / "outputs" / "one_warehouse_multi_retailer" / "learned_benchmark" / run_tag
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(parsed.es_population),
        training_episodes=int(parsed.training_episodes),
        mp_num_processors=1,  # pinned: parallelism is rayon, bounded by RAYON_NUM_THREADS
        save_every=max(1, int(parsed.training_episodes)),
        save_solutions=False,
        horizon=int(reference["benchmark_periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(parsed.train_seed_batch),
        experiment_name=run_tag,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def _get_population_fitness(model, reference, allocation, policy_action_mode):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(p, dtype=np.float32).tolist() for p in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference,
                model,
                flat_params=model.get_model_flat_params(),
                allocation_policy=allocation,
                policy_action_mode=policy_action_mode,
            ).items()
            if key != "flat_params"
        }
        batch_costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            batch_costs.append(
                ir.one_warehouse_multi_retailer_soft_tree_population_rollout(
                    params_batch=params_batch,
                    seeds=[int(s) + seed_offset for s in seeds],
                    **rollout_kwargs,
                )
            )
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [(-float(c), idx) for idx, c in enumerate(costs.tolist())]

    return inner


def _get_model_fitness(model, reference, allocation, policy_action_mode):
    # Fallback single-individual fitness (unused when population fitness is set,
    # but train() requires the callable).
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1, **_kw):
        flat = model.get_model_flat_params() if model_params is None else model_params
        costs = []
        for seed_offset in range(int(getattr(args, "train_seed_batch", 1))):
            c = ir.one_warehouse_multi_retailer_soft_tree_rollout(
                seed=int(seed) + seed_offset,
                **soft_tree_rollout_kwargs(
                    reference, model, flat_params=flat,
                    allocation_policy=allocation, policy_action_mode=policy_action_mode,
                ),
            )
            costs.append(float(c))
        return -float(np.mean(costs)), indiv_idx

    return inner


# --------------------------------------------------------------------------- #
# Per-instance benchmark                                                       #
# --------------------------------------------------------------------------- #
def benchmark_instance(parsed, name: str) -> dict:
    reference = dict(ir.one_warehouse_multi_retailer_get_reference_instance(name))
    symmetric = is_symmetric_retailer_case(reference)
    policy_action_mode = policy_action_mode_for_reference(reference)

    # ---- CRN seed blocks (disjoint search vs held-out) ----
    search_paths = _sample_demand_paths(reference, parsed.search_paths, parsed.search_seed_start)
    holdout_paths = _sample_demand_paths(reference, parsed.holdout_paths, parsed.holdout_seed_start)
    alloc_seed_search = 700_000
    alloc_seed_holdout = 800_000

    # ---- heuristic: grid-search on search block, evaluate on held-out block ----
    heuristics: dict[str, dict] = {}
    for allocation in ("proportional", "min_shortage"):
        searched = _search_best_heuristic_on_paths(
            reference, allocation, search_paths, alloc_seed_search
        )
        holdout_costs = _heuristic_on_paths(
            reference,
            searched["warehouse_base_stock_level"],
            searched["retailer_base_stock_levels"],
            allocation,
            holdout_paths,
            alloc_seed_holdout,
        )
        heuristics[allocation] = {
            "warehouse_base_stock_level": searched["warehouse_base_stock_level"],
            "retailer_base_stock_levels": searched["retailer_base_stock_levels"],
            "search_mean_cost": searched["mean_cost"],
            "holdout_mean_cost": float(holdout_costs.mean()),
            "holdout_std_cost": float(holdout_costs.std()),
            "holdout_stderr": float(holdout_costs.std() / np.sqrt(holdout_costs.size)),
        }

    best_alloc = min(heuristics, key=lambda a: heuristics[a]["holdout_mean_cost"])
    best_heuristic = heuristics[best_alloc]
    best_heuristic_alloc = best_alloc

    # ---- learned soft-tree: train, evaluate on held-out block (same allocation as best heuristic) ----
    model = build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
        policy_action_mode=policy_action_mode,
    )
    train_args = _training_namespace(parsed, reference, model)
    train_start = time.time()
    trained_model, fitness_hist = train(
        model=model,
        get_model_fitness=_get_model_fitness(
            model, reference, parsed.train_allocation, policy_action_mode
        ),
        get_population_fitness=_get_population_fitness(
            model, reference, parsed.train_allocation, policy_action_mode
        ),
        args=train_args,
        same_seed=False,
    )
    train_seconds = time.time() - train_start
    trained_flat = np.asarray(trained_model.get_model_flat_params(), dtype=np.float32).tolist()

    # Evaluate the learned policy under EACH allocation on the held-out block;
    # the headline learned cost uses the same allocation as the best heuristic.
    learned_eval: dict[str, dict] = {}
    for allocation in ("proportional", "min_shortage"):
        costs = _soft_tree_on_paths(
            reference, trained_model, trained_flat, allocation, policy_action_mode,
            holdout_paths, alloc_seed_holdout,
        )
        learned_eval[allocation] = {
            "holdout_mean_cost": float(costs.mean()),
            "holdout_std_cost": float(costs.std()),
            "holdout_stderr": float(costs.std() / np.sqrt(costs.size)),
        }
    learned_best_alloc = min(learned_eval, key=lambda a: learned_eval[a]["holdout_mean_cost"])
    learned_headline = learned_eval[learned_best_alloc]["holdout_mean_cost"]

    # ---- published rows ----
    def pub(row_key: str) -> float | None:
        row = reference.get(row_key)
        return None if row is None else float(-dict(row)["mean_cost"])

    published = {
        "proportional": pub("published_proportional_benchmark"),
        "min_shortage": pub("published_min_shortage_benchmark"),
        "ppo": pub("published_ppo_benchmark"),
    }

    best_heuristic_cost = best_heuristic["holdout_mean_cost"]
    learned_vs_best_heuristic_pct = (
        (best_heuristic_cost - learned_headline) / best_heuristic_cost * 100.0
    )
    learned_vs_ppo_pct = (
        None
        if published["ppo"] is None
        else (published["ppo"] - learned_headline) / published["ppo"] * 100.0
    )
    winner = "learned" if learned_headline < best_heuristic_cost else "heuristic"

    return {
        "instance": name,
        "customer_behavior": reference["customer_behavior"],
        "symmetric_retailers": symmetric,
        "policy_action_mode": policy_action_mode,
        "num_retailers": len(reference["retailer_lead_times"]),
        "periods": int(reference["benchmark_periods"]),
        "heuristics": heuristics,
        "best_heuristic_allocation": best_heuristic_alloc,
        "best_heuristic_cost": best_heuristic_cost,
        "learned": {
            "by_allocation": learned_eval,
            "best_allocation": learned_best_alloc,
            "headline_cost": learned_headline,
            "train_allocation": parsed.train_allocation,
            "train_seconds": train_seconds,
            "best_train_reward": float(np.max(fitness_hist[-1])) if len(fitness_hist) else None,
        },
        "published": published,
        "learned_vs_best_heuristic_pct": learned_vs_best_heuristic_pct,
        "learned_vs_published_ppo_pct": learned_vs_ppo_pct,
        "winner_learned_vs_heuristic": winner,
        "trained_flat_params": trained_flat,
        "tree_config": {
            "depth": parsed.depth,
            "temperature": parsed.temperature,
            "split_type": parsed.split_type,
            "leaf_type": parsed.leaf_type,
            "training_episodes": parsed.training_episodes,
            "es_population": parsed.es_population,
            "sigma_init": parsed.sigma_init,
            "train_seed_batch": parsed.train_seed_batch,
            "seed": parsed.seed,
        },
    }


def _result_line(row: dict) -> str:
    learned = row["learned"]["headline_cost"]
    heur = row["best_heuristic_cost"]
    ppo = row["published"]["ppo"]
    ppo_str = f"{ppo:.1f}" if ppo is not None else "n/a"
    win = row["winner_learned_vs_heuristic"]
    gap = row["learned_vs_best_heuristic_pct"]
    return (
        f"{row['instance']} ({row['customer_behavior']}): "
        f"learned {learned:.1f} ({row['learned']['best_allocation']}) vs "
        f"heuristic {heur:.1f} ({row['best_heuristic_allocation']}) vs "
        f"published-PPO {ppo_str} -> {win} wins, {gap:+.2f}% vs best heuristic"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--instance_names",
        nargs="+",
        default=["kaynov2024_instance_1", "kaynov2024_instance_6", "kaynov2024_instance_11"],
    )
    p.add_argument("--train_allocation", default="proportional",
                   choices=["proportional", "min_shortage", "random_sequential"])
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--training_episodes", type=int, default=400)
    p.add_argument("--es_population", type=int, default=24)
    p.add_argument("--sigma_init", type=float, default=1.5)
    p.add_argument("--train_seed_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--search_paths", type=int, default=256)
    p.add_argument("--holdout_paths", type=int, default=4096)
    p.add_argument("--search_seed_start", type=int, default=500000)
    p.add_argument("--holdout_seed_start", type=int, default=900000)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    parsed = parse_args()

    # exact-DP self-consistency anchor (env transition/cost validation)
    dp = dict(ir.one_warehouse_multi_retailer_exact_dp_summary())
    exact_check = {
        "optimal_discounted_cost": dp["optimal_discounted_cost"],
        "proportional_discounted_cost": dp["proportional_discounted_cost"],
        "min_shortage_discounted_cost": dp["min_shortage_discounted_cost"],
        "optimal_dominates_proportional": dp["optimal_discounted_cost"]
        <= dp["proportional_discounted_cost"] + 1e-9,
        "optimal_dominates_min_shortage": dp["optimal_discounted_cost"]
        <= dp["min_shortage_discounted_cost"] + 1e-9,
    }

    rows = []
    for name in parsed.instance_names:
        print(f"\n=== {name} ===", flush=True)
        row = benchmark_instance(parsed, name)
        rows.append(row)
        print(_result_line(row), flush=True)

    payload = {
        "family": "one_warehouse_multi_retailer",
        "source": "Kaynov et al. (2024), IJPE 267, 109088",
        "protocol": {
            "periods": 100,
            "discount_factor": DISCOUNT_FACTOR,
            "initial_state_rule": "mean_filled_pipeline_warm_start",
            "evaluation": "CRN held-out demand paths via *_from_paths bindings; "
            "heuristic and learned scored on identical paths",
            "search_paths": parsed.search_paths,
            "holdout_paths": parsed.holdout_paths,
            "search_seed_start": parsed.search_seed_start,
            "holdout_seed_start": parsed.holdout_seed_start,
            "train_allocation": parsed.train_allocation,
            "cpu_cap": "RAYON_NUM_THREADS bounds rayon; mp_num_processors=1 (no Python pool)",
            "tree_config": {
                "depth": parsed.depth,
                "temperature": parsed.temperature,
                "split_type": parsed.split_type,
                "leaf_type": parsed.leaf_type,
                "training_episodes": parsed.training_episodes,
                "es_population": parsed.es_population,
                "train_seed_batch": parsed.train_seed_batch,
                "sigma_init": parsed.sigma_init,
            },
        },
        "exact_dp_self_consistency": exact_check,
        "rows": rows,
        "result_lines": [_result_line(r) for r in rows],
    }

    if parsed.output_json:
        out = Path(parsed.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}", flush=True)

    print("\n--- SUMMARY ---")
    for line in payload["result_lines"]:
        print(line)
    print("\nexact DP self-consistency:", json.dumps(exact_check))


if __name__ == "__main__":
    main()
