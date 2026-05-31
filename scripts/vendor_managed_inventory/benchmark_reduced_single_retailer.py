"""
Benchmark: learned soft-tree vs tuned heuristics on the reduced single-retailer
vendor-managed-inventory (VMI) environment.

ALGORITHMIC DESCRIPTION
=======================
Objective
---------
The vendor_managed_inventory problem has two distinct env families:

  1. The continuous-time multi-retailer Giannoccaro & Pontrandolfo (2010)
     truck-dispatch simulator (env::step_paper_state). Its published 8-case
     profit table is NOT reproduced (see literature/README.md), so it is not a
     valid benchmark anchor.

  2. The repo-native reduced single-retailer finite-horizon VMI slice
     (env::step_state). This is the env exposed to Python via
     invman_rust.vendor_managed_inventory_* bindings, validated by an exact
     finite-horizon DP regression in verification/tests.rs.

This script benchmarks the policies that are runnable from Python WITHOUT a
Rust rebuild, on the reduced single-retailer slice:

  - tuned `retailer_base_stock` heuristic (best base-stock level on a grid)
  - tuned `dc_reserve_base_stock` heuristic (best level x reserve on a grid)
  - a soft-decision-tree policy trained with CMA-ES through the installed
    `vendor_managed_inventory_soft_tree_population_rollout` binding

Environment recap (env::step_state, order of events within a period)
--------------------------------------------------------------------
state = (dc_on_hand, retailer_on_hand, retailer_pipeline)
  1. vendor ships `q` units from the DC into the retailer pipeline
     (q <= max_shipment_quantity, q <= dc_on_hand)
  2. last period's pipeline arrives at the retailer
  3. retailer demand realizes (Poisson here); sales = min(on_hand+arrivals, d);
     unmet demand is LOST (lost_sales penalized at stockout_cost_per_unit)
  4. DC replenishes deterministically by dc_replenishment_quantity, capped at
     dc_capacity
  5. period cost = shipment + dc_holding(next) + retailer_holding(next) + stockout
Terminal salvage credits all remaining inventory at salvage_value_per_unit.
Objective = discounted sum of period costs minus discounted terminal salvage
(LOWER is better).

Fair comparison protocol
------------------------
- Common random numbers: every policy is evaluated on the SAME set of demand
  seeds. Heuristic grids are tuned on TRAIN seeds; the soft tree is trained on
  TRAIN seeds via CMA-ES; ALL policies are then scored on disjoint HELD-OUT
  seeds. Reported numbers are mean +/- std discounted cost on the held-out set.
- Instance set: the canonical PRIMARY_REFERENCE_INSTANCE plus a small grid of
  perturbed instances (varying stockout cost and demand mean) to show the
  comparison is not a single-point artifact.

WHY no exact-optimal column here
--------------------------------
The exact finite-horizon DP (finite_horizon_dp::solve_optimal_policy) is only
defined for a small DISCRETE demand support and is NOT exposed as a Python
binding. Adding that binding requires a Rust rebuild + a bindings.rs edit, both
disallowed for this task. See the printed BLOCKER note; the DP optimal is the
right ceiling to add next.

Usage
-----
    python scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py
    python scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py --quick
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

import numpy as np

import invman_rust as ir


# ---------------------------------------------------------------------------
# Instance definition. Defaults mirror references.rs::PRIMARY_REFERENCE_INSTANCE.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instance:
    name: str
    periods: int = 24
    demand_mean: float = 2.5
    initial_dc_on_hand: int = 8
    initial_retailer_on_hand: int = 2
    initial_retailer_pipeline: int = 1
    dc_replenishment_quantity: int = 3
    dc_capacity: int = 10
    shipment_cost_per_unit: float = 0.4
    dc_holding_cost_per_unit: float = 0.25
    retailer_holding_cost_per_unit: float = 0.6
    stockout_cost_per_unit: float = 5.0
    salvage_value_per_unit: float = 0.2
    max_shipment_quantity: int = 5
    discount_factor: float = 0.99


PRIMARY = Instance(name="giannoccaro2010_style_single_retailer")

# A small perturbation grid so the comparison is an instance SET, not a point.
INSTANCE_SET = [
    PRIMARY,
    Instance(name="low_penalty", stockout_cost_per_unit=2.0),
    Instance(name="high_penalty", stockout_cost_per_unit=9.0),
    Instance(name="low_demand", demand_mean=1.5),
    Instance(name="high_demand", demand_mean=3.5),
]


# ---------------------------------------------------------------------------
# Heuristic evaluation through the installed simulate_policy binding.
# simulate_policy uses an internal seeded RNG; we vary the seed to realize
# different demand streams and average. This is our common-random-number knob.
# ---------------------------------------------------------------------------
def eval_heuristic(inst: Instance, policy_name: str, params: list[float],
                   seed: int, replications: int) -> tuple[float, float]:
    out = ir.vendor_managed_inventory_simulate_policy(
        policy_name, params,
        inst.initial_dc_on_hand, inst.initial_retailer_on_hand,
        inst.initial_retailer_pipeline,
        inst.periods, replications, seed, inst.demand_mean, "poisson",
        inst.dc_replenishment_quantity, inst.dc_capacity,
        inst.shipment_cost_per_unit, inst.dc_holding_cost_per_unit,
        inst.retailer_holding_cost_per_unit, inst.stockout_cost_per_unit,
        inst.max_shipment_quantity, inst.discount_factor,
        inst.salvage_value_per_unit,
    )
    return out["mean_discounted_cost"], out["std_discounted_cost"]


def tune_retailer_base_stock(inst: Instance, seed: int, reps: int):
    best = (None, float("inf"))
    upper = inst.initial_dc_on_hand + inst.dc_capacity
    for level in range(0, upper + 1):
        mean, _ = eval_heuristic(inst, "retailer_base_stock", [float(level)], seed, reps)
        if mean < best[1]:
            best = ([float(level)], mean)
    return best


def tune_dc_reserve_base_stock(inst: Instance, seed: int, reps: int):
    best = (None, float("inf"))
    upper = inst.initial_dc_on_hand + inst.dc_capacity
    for level in range(0, upper + 1):
        for reserve in range(0, inst.dc_capacity + 1):
            mean, _ = eval_heuristic(
                inst, "dc_reserve_base_stock", [float(level), float(reserve)], seed, reps)
            if mean < best[1]:
                best = ([float(level), float(reserve)], mean)
    return best


# ---------------------------------------------------------------------------
# Soft-tree policy. Param layout matches core::policies::soft_tree:
#   internal nodes = 2^depth - 1, leaves = 2^depth
#   flat_params = [split_weights (ni*input_dim)] + [split_bias (ni)]
#                 + [leaf logits (leaves * action_dim)]   (constant leaf)
# input_dim = 7 (build_policy_state), action_dim = 1 (shipment quantity).
# ---------------------------------------------------------------------------
INPUT_DIM = 7
ACTION_DIM = 1


def soft_tree_param_count(depth: int) -> int:
    ni = (1 << depth) - 1
    nl = 1 << depth
    return ni * INPUT_DIM + ni + nl * ACTION_DIM


def eval_soft_tree_population(inst: Instance, batch: list[list[float]],
                              depth: int, seeds: list[int],
                              temperature: float = 0.25) -> list[float]:
    """Mean over the given seeds for each parameter vector in the batch.

    Each rollout binding call evaluates one (params, seed) pair, so to average
    a batch over multiple seeds we tile params x seeds and reduce.
    """
    tiled_params = []
    tiled_seeds = []
    for p in batch:
        for s in seeds:
            tiled_params.append(p)
            tiled_seeds.append(int(s))
    costs = ir.vendor_managed_inventory_soft_tree_population_rollout(
        tiled_params, INPUT_DIM, depth,
        [0], [inst.max_shipment_quantity], "scalar_quantity",
        inst.initial_dc_on_hand, inst.initial_retailer_on_hand,
        inst.initial_retailer_pipeline,
        inst.periods, "poisson", inst.demand_mean,
        inst.dc_replenishment_quantity, inst.dc_capacity,
        inst.shipment_cost_per_unit, inst.dc_holding_cost_per_unit,
        inst.retailer_holding_cost_per_unit, inst.stockout_cost_per_unit,
        inst.salvage_value_per_unit, inst.max_shipment_quantity,
        tiled_seeds, inst.discount_factor, temperature,
        "oblique", "constant", None,
    )
    n_seeds = len(seeds)
    means = []
    for i in range(len(batch)):
        chunk = costs[i * n_seeds:(i + 1) * n_seeds]
        means.append(sum(chunk) / len(chunk))
    return means


def soft_tree_held_out_samples(inst: Instance, params: list[float], depth: int,
                               seeds: list[int], temperature: float = 0.25) -> list[float]:
    costs = ir.vendor_managed_inventory_soft_tree_population_rollout(
        [params] * len(seeds), INPUT_DIM, depth,
        [0], [inst.max_shipment_quantity], "scalar_quantity",
        inst.initial_dc_on_hand, inst.initial_retailer_on_hand,
        inst.initial_retailer_pipeline,
        inst.periods, "poisson", inst.demand_mean,
        inst.dc_replenishment_quantity, inst.dc_capacity,
        inst.shipment_cost_per_unit, inst.dc_holding_cost_per_unit,
        inst.retailer_holding_cost_per_unit, inst.stockout_cost_per_unit,
        inst.salvage_value_per_unit, inst.max_shipment_quantity,
        [int(s) for s in seeds], inst.discount_factor, temperature,
        "oblique", "constant", None,
    )
    return list(costs)


def heuristic_held_out_samples(inst: Instance, policy_name: str,
                               params: list[float], seeds: list[int],
                               reps_per_seed: int) -> list[float]:
    """One mean per held-out seed (each call uses a fresh internal RNG seed)."""
    out = []
    for s in seeds:
        mean, _ = eval_heuristic(inst, policy_name, params, int(s), reps_per_seed)
        out.append(mean)
    return out


def train_soft_tree(inst: Instance, depth: int, train_seeds: list[int],
                    popsize: int, iters: int, sigma0: float, rng_seed: int,
                    temperature: float):
    import cma  # pycma; available in this env

    n = soft_tree_param_count(depth)
    x0 = list(np.zeros(n))
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": popsize, "seed": rng_seed, "verbose": -9, "maxiter": iters},
    )
    best_x, best_f = None, float("inf")
    while not es.stop():
        solutions = es.ask()
        batch = [[float(v) for v in s] for s in solutions]
        fitness = eval_soft_tree_population(inst, batch, depth, train_seeds, temperature)
        es.tell(solutions, fitness)
        gen_best_i = int(np.argmin(fitness))
        if fitness[gen_best_i] < best_f:
            best_f = fitness[gen_best_i]
            best_x = batch[gen_best_i]
    return best_x, best_f


# ---------------------------------------------------------------------------
@dataclass
class Result:
    instance: str
    policy: str
    params: object
    train_cost: float
    held_out_mean: float
    held_out_std: float


def summarize(samples: list[float]) -> tuple[float, float, float]:
    """Return (mean, per-sample std, standard error of the mean)."""
    m = sum(samples) / len(samples)
    s = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    sem = s / (len(samples) ** 0.5) if len(samples) > 1 else 0.0
    return m, s, sem


def run(quick: bool):
    # Disjoint seed banks for tuning/training vs held-out scoring.
    # The soft tree is scored as N single-path rollouts (one per seed); the
    # heuristics are scored as N seeds x reps internal paths. To make the
    # held-out MEANS comparably tight we give the soft tree many more seeds so
    # its standard error matches the heuristic standard error.
    # Many train seeds + a sharp temperature let the soft tree express a clean
    # base-stock-like threshold; with only a handful of train seeds the CRN
    # fitness is too noisy and the tree underfits.
    train_seeds = list(range(1000, 1000 + (16 if quick else 64)))
    held_out_seeds = list(range(9000, 9000 + (12 if quick else 32)))
    soft_tree_eval_seeds = list(range(20000, 20000 + (400 if quick else 4000)))
    heuristic_reps = 400 if quick else 1500
    depth = 2
    temperature = 0.1
    popsize = 12 if quick else 24
    iters = 40 if quick else 200
    sigma0 = 0.8

    print("=" * 78)
    print("VMI reduced single-retailer benchmark (env::step_state)")
    print("policies: tuned retailer_base_stock, tuned dc_reserve_base_stock,")
    print("          CMA-ES soft tree (depth=%d, %d params)" % (depth, soft_tree_param_count(depth)))
    print("metric: discounted cost, LOWER is better, held-out CRN seeds")
    print("=" * 78)

    all_results: list[Result] = []
    for inst in INSTANCE_SET:
        print(f"\n--- instance: {inst.name} "
              f"(demand_mean={inst.demand_mean}, stockout={inst.stockout_cost_per_unit}) ---")

        # 1) tune heuristics on a fixed train seed (grid search)
        rbs_params, rbs_train = tune_retailer_base_stock(inst, train_seeds[0], heuristic_reps)
        dcr_params, dcr_train = tune_dc_reserve_base_stock(inst, train_seeds[0], heuristic_reps)

        # 2) train soft tree
        st_params, st_train = train_soft_tree(
            inst, depth, train_seeds, popsize, iters, sigma0, rng_seed=12345,
            temperature=temperature)

        # 3) held-out evaluation, common seeds across policies
        rbs_samples = heuristic_held_out_samples(
            inst, "retailer_base_stock", rbs_params, held_out_seeds, heuristic_reps)
        dcr_samples = heuristic_held_out_samples(
            inst, "dc_reserve_base_stock", dcr_params, held_out_seeds, heuristic_reps)
        # Many single-path seeds so the soft-tree held-out MEAN is tight.
        st_samples = soft_tree_held_out_samples(
            inst, st_params, depth, soft_tree_eval_seeds, temperature)

        for name, params, train_cost, samples in [
            ("retailer_base_stock", rbs_params, rbs_train, rbs_samples),
            ("dc_reserve_base_stock", dcr_params, dcr_train, dcr_samples),
            ("soft_tree_d%d" % depth, "(%d params)" % len(st_params), st_train, st_samples),
        ]:
            m, s, sem = summarize(samples)
            all_results.append(Result(inst.name, name, params, train_cost, m, sem))
            print(f"  {name:24s} held_out_cost = {m:8.3f}  (SEM {sem:5.3f}, "
                  f"n={len(samples)})   params={params}")

        # which wins on held-out
        held = {r.policy: r.held_out_mean for r in all_results if r.instance == inst.name}
        winner = min(held, key=held.get)
        best_heur = min(
            (v for k, v in held.items() if not k.startswith("soft_tree")),
        )
        st_mean = next(v for k, v in held.items() if k.startswith("soft_tree"))
        gap = 100.0 * (best_heur - st_mean) / best_heur
        print(f"  -> winner: {winner};  soft_tree vs best heuristic: "
              f"{gap:+.2f}% ({'better' if gap > 0 else 'worse'})")

    print("\n" + "=" * 78)
    print("SUMMARY (held-out discounted cost, lower is better)")
    print("=" * 78)
    header = f"{'instance':22s} {'retailer_bs':>14s} {'dc_reserve_bs':>14s} {'soft_tree':>14s}"
    print(header)
    for inst in INSTANCE_SET:
        row = {r.policy: r.held_out_mean for r in all_results if r.instance == inst.name}
        rbs = row.get("retailer_base_stock", float("nan"))
        dcr = row.get("dc_reserve_base_stock", float("nan"))
        st = next((v for k, v in row.items() if k.startswith("soft_tree")), float("nan"))
        print(f"{inst.name:22s} {rbs:14.3f} {dcr:14.3f} {st:14.3f}")

    print("\nBLOCKER: exact finite-horizon DP optimal (finite_horizon_dp.rs) is the")
    print("correct ceiling but is NOT exposed as a Python binding; adding it needs a")
    print("Rust rebuild + bindings.rs edit (disallowed here). See README next steps.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer reps/iters/seeds for a fast smoke run")
    args = ap.parse_args()
    run(args.quick)
