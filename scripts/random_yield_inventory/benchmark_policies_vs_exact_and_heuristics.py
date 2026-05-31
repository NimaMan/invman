"""
benchmark_policies_vs_exact_and_heuristics.py
=============================================

OBJECTIVE
---------
Benchmark the random_yield_inventory problem at the same standard as the repo's
finished problems: compare the available policies (exact-optimal DP, the two
literature heuristics LIR and WNH, and -- optionally -- a CMA-ES-trained soft-tree
learned policy) against each other on the repo's reference instances, and report
concrete numbers.

This script is SELF-CONTAINED against the *currently installed* invman_rust and the
*current* invman.policy.Policy / invman.es_mp.train interface. It deliberately does
NOT import the stale `invman.policies.soft_tree` module that the older scripts
(common.py / train_soft_tree_reference.py) reference -- that module path no longer
exists in the package, which is why those scripts cannot run.

ALGORITHM / WHAT IS COMPARED
----------------------------
Two benchmark slices are produced.

1. EXACT-DP SLICE (verified-correct, capped action space).
   On the reduced VERIFICATION_PROBLEM_INSTANCE (finite horizon, discrete demand,
   lead time 2, action space capped at max_order_quantity), the Rust exact DP
   (finite_horizon_dp.rs) enumerates the full (demand x all-or-nothing-yield) tree
   and returns:
     - optimal discounted cost + first action
     - LIR (linear_inflation) discounted cost evaluated under the SAME exact DP
     - WNH (weighted_newsvendor) discounted cost evaluated under the SAME exact DP
   Optimality gaps of each heuristic to the exact optimum are reported. The exact DP
   has been validated against an independent Python re-implementation of the same MDP
   (cost 40.0598976099, first action 4), so this slice is an implementation-correct
   anchor -- NOT a literature-verified one (no public per-instance number exists in
   Yan 2026 / Chen 2018 to assert against).

   IMPORTANT timing/cap note: the exact DP CLAMPS the heuristic order to
   max_order_quantity (a DP-tractability truncation, not a physical cap). The
   uncapped env (heuristics/mod.rs) does not clamp, so a Monte-Carlo env rollout of a
   heuristic that wants to order > cap will read slightly higher than the capped DP
   value. This is expected and benign; the optimal policy does not bind the cap.

2. SIMULATION SLICE (primary reference instance, uncapped env).
   On PRIMARY_REFERENCE_INSTANCE (Poisson demand, horizon 12) the LIR and WNH
   heuristics are simulated over a shared seed set via the Rust env rollout. If
   --train_soft_tree is passed, a soft-tree policy is trained with CMA-ES against the
   Rust rollout and evaluated on the same seed set; otherwise only the heuristics are
   reported. Mean discounted cost, std, and pairwise gaps are printed.

USAGE
-----
    python scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py
    python scripts/random_yield_inventory/benchmark_policies_vs_exact_and_heuristics.py \
        --train_soft_tree --training_episodes 400 --es_population 24 --output_json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust  # noqa: E402
from invman.policy import Policy  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference instances (source of truth = Rust literature/references.rs)        #
# --------------------------------------------------------------------------- #
def get_primary_reference() -> dict:
    return dict(invman_rust.random_yield_inventory_primary_reference_instance())


def get_exact_dp_summary() -> dict:
    return dict(invman_rust.random_yield_inventory_exact_dp_summary())


def linear_inflation_params(ref: dict) -> list[float]:
    target, factor = invman_rust.random_yield_inventory_linear_inflation_parameters(
        demand_mean=float(ref["demand_mean"]),
        success_probability=float(ref["success_probability"]),
        lead_time=int(ref["lead_time"]),
        holding_cost=float(ref["holding_cost"]),
        shortage_cost=float(ref["shortage_cost"]),
    )
    return [float(target), float(factor)]


# --------------------------------------------------------------------------- #
# Heuristic simulation on the primary (uncapped) instance                      #
# --------------------------------------------------------------------------- #
def evaluate_heuristic(ref: dict, name: str, params: list[float], seeds: list[int]) -> dict:
    return dict(
        invman_rust.random_yield_inventory_policy_discounted_cost_summary(
            policy_name=name,
            params=[float(v) for v in params],
            initial_inventory_level=float(ref["initial_inventory_level"]),
            pipeline_orders=[float(v) for v in ref["initial_pipeline_orders"]],
            periods=int(ref["periods"]),
            seeds=[int(s) for s in seeds],
            demand_mean=float(ref["demand_mean"]),
            success_probability=float(ref["success_probability"]),
            holding_cost=float(ref["holding_cost"]),
            shortage_cost=float(ref["shortage_cost"]),
            procurement_cost=float(ref["procurement_cost"]),
            discount_factor=float(ref["discount_factor"]),
            demand_distribution="poisson",
        )
    )


# --------------------------------------------------------------------------- #
# Soft-tree learned policy (current Policy interface, Rust rollout)            #
# --------------------------------------------------------------------------- #
def _soft_tree_rollout_kwargs(ref: dict, cap: int, depth: int, temperature: float,
                              split_type: str, leaf_type: str, flat) -> dict:
    return dict(
        flat_params=np.asarray(flat, dtype=np.float32).tolist(),
        input_dim=int(ref["lead_time"]) + 3,
        depth=int(depth),
        min_values=[0],
        max_values=[int(cap)],
        action_mode="scalar_quantity",
        initial_inventory_level=float(ref["initial_inventory_level"]),
        pipeline_orders=[float(v) for v in ref["initial_pipeline_orders"]],
        periods=int(ref["periods"]),
        demand_mean=float(ref["demand_mean"]),
        success_probability=float(ref["success_probability"]),
        holding_cost=float(ref["holding_cost"]),
        shortage_cost=float(ref["shortage_cost"]),
        procurement_cost=float(ref["procurement_cost"]),
        discount_factor=float(ref["discount_factor"]),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        allowed_values=None,
    )


def train_soft_tree(ref: dict, *, cap: int, depth: int, temperature: float,
                    split_type: str, leaf_type: str, episodes: int, population: int,
                    sigma_init: float, seed: int, train_seed_batch: int) -> Policy:
    from invman.es_mp import train

    model = Policy(
        backbone="soft_tree",
        input_dim=int(ref["lead_time"]) + 3,
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(int(cap),),
        depth=int(depth),
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        state_normalizer="identity",
    )

    def get_model_fitness(m, args, model_params=None, seed=1234, indiv_idx=-1,
                          return_env=False, track_demand=False, verbose=False):
        del return_env, track_demand, verbose
        flat = m.get_model_flat_params() if model_params is None else model_params
        costs = []
        for off in range(int(getattr(args, "train_seed_batch", 1))):
            costs.append(invman_rust.random_yield_inventory_soft_tree_rollout(
                seed=int(seed) + off, demand_distribution="poisson",
                **_soft_tree_rollout_kwargs(ref, cap, depth, temperature, split_type, leaf_type, flat)))
        return -float(np.mean(costs)), indiv_idx

    def get_population_fitness(m, args, model_params_batch, seeds):
        batch = [np.asarray(p, dtype=np.float32).tolist() for p in model_params_batch]
        kw = {k: v for k, v in _soft_tree_rollout_kwargs(
            ref, cap, depth, temperature, split_type, leaf_type, m.get_model_flat_params()).items()
            if k != "flat_params"}
        acc = []
        for off in range(int(getattr(args, "train_seed_batch", 1))):
            acc.append(invman_rust.random_yield_inventory_soft_tree_population_rollout(
                params_batch=batch, seeds=[int(s) + off for s in seeds],
                demand_distribution="poisson", **kw))
        costs = np.mean(np.asarray(acc, dtype=np.float64), axis=0)
        return [(-float(c), i) for i, c in enumerate(costs.tolist())]

    args = SimpleNamespace(
        training_method="cma", sigma_init=float(sigma_init), es_population=int(population),
        training_episodes=int(episodes), mp_num_processors=1, save_every=10 ** 9,
        save_solutions=False, horizon=int(ref["periods"]), seed=int(seed),
        train_seed_batch=int(train_seed_batch), experiment_name="random_yield_benchmark",
        log_dir="/tmp/random_yield_benchmark/logs",
        trained_models_dir="/tmp/random_yield_benchmark/models",
    )
    trained, _ = train(model=model, get_model_fitness=get_model_fitness,
                        get_population_fitness=get_population_fitness, args=args, same_seed=False)
    return trained


def evaluate_soft_tree(ref: dict, model: Policy, *, cap: int, depth: int,
                       temperature: float, split_type: str, leaf_type: str,
                       seeds: list[int]) -> dict:
    costs = [
        invman_rust.random_yield_inventory_soft_tree_rollout(
            seed=int(s), demand_distribution="poisson",
            **_soft_tree_rollout_kwargs(ref, cap, depth, temperature, split_type, leaf_type,
                                        model.get_model_flat_params()))
        for s in seeds
    ]
    costs = np.asarray(costs, dtype=np.float64)
    return {"mean_cost": float(costs.mean()), "cost_std": float(costs.std()),
            "min_cost": float(costs.min()), "max_cost": float(costs.max()),
            "num_samples": int(costs.size)}


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_seeds", type=int, default=2000)
    p.add_argument("--train_soft_tree", action="store_true")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--training_episodes", type=int, default=400)
    p.add_argument("--es_population", type=int, default=24)
    p.add_argument("--sigma_init", type=float, default=1.5)
    p.add_argument("--train_seed_batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--action_cap", type=int, default=32)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    a = parse_args()

    # ---- Slice 1: exact-DP (verified-correct, capped) ---------------------- #
    exact = get_exact_dp_summary()
    exact_rows = [
        {"policy": "exact_optimal_dp", "cost": exact["optimal_discounted_cost"],
         "first_action": exact["optimal_first_action"], "gap_to_optimal": 0.0},
        {"policy": "linear_inflation", "cost": exact["linear_inflation_discounted_cost"],
         "first_action": exact["linear_inflation_first_action"],
         "gap_to_optimal": exact["linear_inflation_gap_to_optimal"]},
        {"policy": "weighted_newsvendor", "cost": exact["weighted_newsvendor_discounted_cost"],
         "first_action": exact["weighted_newsvendor_first_action"],
         "gap_to_optimal": exact["weighted_newsvendor_gap_to_optimal"]},
    ]

    # ---- Slice 2: simulation on the primary (uncapped) instance ------------ #
    ref = get_primary_reference()
    seeds = list(range(123, 123 + int(a.eval_seeds)))
    lir_params = linear_inflation_params(ref)
    lir = evaluate_heuristic(ref, "linear_inflation", lir_params, seeds)
    wnh = evaluate_heuristic(ref, "weighted_newsvendor", [], seeds)

    sim_rows = [
        {"policy": "linear_inflation", "params": str(lir_params),
         "mean_cost": lir["mean_cost"], "cost_std": lir["cost_std"]},
        {"policy": "weighted_newsvendor", "params": "[]",
         "mean_cost": wnh["mean_cost"], "cost_std": wnh["cost_std"]},
    ]

    soft_tree_eval = None
    if a.train_soft_tree:
        trained = train_soft_tree(
            ref, cap=a.action_cap, depth=a.depth, temperature=a.temperature,
            split_type=a.split_type, leaf_type=a.leaf_type, episodes=a.training_episodes,
            population=a.es_population, sigma_init=a.sigma_init, seed=a.seed,
            train_seed_batch=a.train_seed_batch)
        soft_tree_eval = evaluate_soft_tree(
            ref, trained, cap=a.action_cap, depth=a.depth, temperature=a.temperature,
            split_type=a.split_type, leaf_type=a.leaf_type, seeds=seeds)
        sim_rows.append({"policy": f"soft_tree(d={a.depth},{a.leaf_type})",
                         "params": f"trained {a.training_episodes}ep pop{a.es_population}",
                         "mean_cost": soft_tree_eval["mean_cost"],
                         "cost_std": soft_tree_eval["cost_std"]})

    best_sim = min(r["mean_cost"] for r in sim_rows)
    for r in sim_rows:
        r["gap_to_best"] = float(r["mean_cost"] - best_sim)

    payload = {
        "literature_verified": False,
        "exact_dp_slice": {
            "instance": "VERIFICATION_PROBLEM_INSTANCE (capped, discrete demand, L=2)",
            "validated_against": "independent python DP: cost 40.0598976099, first_action 4",
            "rows": exact_rows,
        },
        "simulation_slice": {
            "instance": ref["name"],
            "periods": ref["periods"], "num_seeds": len(seeds),
            "rows": sim_rows,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    # markdown
    print("\n### Exact-DP slice (capped, implementation-verified optimum)\n")
    print("| Policy | Discounted Cost | First Action | Gap to Optimal |")
    print("| --- | ---: | ---: | ---: |")
    for r in exact_rows:
        print(f"| `{r['policy']}` | {r['cost']:.4f} | {r['first_action']} | {r['gap_to_optimal']:.4f} |")
    print(f"\n### Simulation slice ({ref['name']}, {len(seeds)} seeds, uncapped env)\n")
    print("| Policy | Params | Mean Discounted Cost | Std | Gap to Best |")
    print("| --- | --- | ---: | ---: | ---: |")
    for r in sim_rows:
        print(f"| `{r['policy']}` | `{r['params']}` | {r['mean_cost']:.3f} | "
              f"{r['cost_std']:.3f} | {r['gap_to_best']:.3f} |")

    if a.output_json:
        out = Path(a.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
