"""
Self-contained heuristic-vs-published benchmark for one_warehouse_multi_retailer.

OBJECTIVE
---------
Quantify, with numbers reproducible against the *currently installed* `invman_rust`,
how the repo's echelon base-stock + allocation heuristics compare to the published
Kaynov et al. (2024) Table A.3 benchmark rows on the carried instance set.

Why this script exists separately from `run_paper_benchmark.py`:
  - `run_paper_benchmark.py` / `common.py` import `invman.policies.soft_tree.SoftTreePolicy`,
    a module path that no longer exists after the repo's policy refactor. That makes the
    *learned soft-tree* path uninstantiable today. The heuristic + exact-DP comparison, by
    contrast, needs only `invman_rust`, so it is isolated here and runs as-is.

ALGORITHM (per instance)
-------------------------
1. Build the benchmark initial state with the "mean-filled pipeline warm start" rule:
   warehouse inventory and every pipeline slot are seeded with the rounded one-period
   mean demand. (This matches the rule used by scripts/.../common.py:benchmark_initial_state.)
2. Derive an echelon base-stock search grid from one-period demand moments
   (mean*lead_periods +/- 3*std*sqrt(lead_periods)); collapse to a 1-D retailer grid when
   the instance is symmetric.
3. For each allocation rule in {proportional, min_shortage}: grid-search the warehouse and
   retailer base-stock levels at `--search_replications` trajectories, then re-evaluate the
   argmin at `--eval_replications` trajectories with a fresh seed. Costs are undiscounted
   100-period totals (discount_factor = 1.0), matching the paper's evaluation horizon.
4. Compare repo cost to the published cost (= -published reward; the paper reports negative
   reward, the repo reports positive cost).
5. Cross-check the env transition/cost against the repo-native exact finite-horizon DP on the
   reduced VERIFICATION_PROBLEM_INSTANCE (optimal <= both heuristics).

HONEST STATUS
-------------
The repo reproduces the *shape* and *order of magnitude* of the published heuristics but does
NOT bit-match them: on the symmetric instances the repo lands ~1-5% BELOW the published cost
(systematic, same direction), attributable to the warm-start initial condition and the
repo-defined search grid rather than a transition bug. See literature/README.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import invman_rust as ir


# --------------------------------------------------------------------------- moments

def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0.0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def _rounded_normal_moments(mean: float, std: float) -> tuple[float, float]:
    if std <= 0.0:
        clipped = max(int(round(mean)), 0)
        return float(clipped), 0.0
    probs = [max(0.0, min(1.0, _normal_cdf(0.5, mean, std)))]
    support = [0]
    k = 1
    cumulative = probs[0]
    while cumulative < 1.0 - 1e-12 and k < 10_000:
        prob = max(0.0, _normal_cdf(k + 0.5, mean, std) - _normal_cdf(k - 0.5, mean, std))
        if prob > 1e-15:
            probs.append(prob)
            support.append(k)
            cumulative += prob
        k += 1
    if cumulative < 1.0:
        probs[-1] += 1.0 - cumulative
    sup = np.asarray(support, dtype=np.float64)
    prob_arr = np.asarray(probs, dtype=np.float64)
    mean_value = float((sup * prob_arr).sum())
    variance = float((((sup - mean_value) ** 2) * prob_arr).sum())
    return mean_value, math.sqrt(max(variance, 0.0))


def demand_moments(reference: dict) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for kind, p1, p2 in zip(
        reference["demand_kinds"], reference["demand_param1"], reference["demand_param2"]
    ):
        if kind == "poisson":
            means.append(float(p1))
            stds.append(math.sqrt(float(p1)))
        elif kind == "discrete_uniform":
            low = int(round(p1))
            high = int(round(p2))
            n = high - low + 1
            means.append(0.5 * (low + high))
            stds.append(math.sqrt((n * n - 1) / 12.0))
        elif kind == "rounded_normal":
            mean_value, std_value = _rounded_normal_moments(float(p1), float(p2))
            means.append(mean_value)
            stds.append(std_value)
        else:  # deterministic
            means.append(float(p1))
            stds.append(0.0)
    return means, stds


# --------------------------------------------------------------------------- state / bounds

def initial_state(reference: dict) -> tuple[int, list[int], list[int], list[list[int]]]:
    means, _ = demand_moments(reference)
    warehouse_mean = int(round(sum(means)))
    retailer_inventory = [int(round(m)) for m in means]
    warehouse_pipeline = [warehouse_mean] * int(reference["warehouse_lead_time"])
    retailer_pipeline = [
        [retailer_inventory[i]] * int(lt)
        for i, lt in enumerate(reference["retailer_lead_times"])
    ]
    return warehouse_mean, warehouse_pipeline, retailer_inventory, retailer_pipeline


def is_symmetric(reference: dict) -> bool:
    return (
        len(set(reference["retailer_lead_times"])) == 1
        and len(set(reference["holding_cost_retailers"])) == 1
        and len(set(reference["penalty_costs_retailers"])) == 1
        and all(
            reference["demand_kinds"][i] == reference["demand_kinds"][0]
            and reference["demand_param1"][i] == reference["demand_param1"][0]
            and reference["demand_param2"][i] == reference["demand_param2"][0]
            for i in range(1, len(reference["retailer_lead_times"]))
        )
    )


def search_bounds(reference: dict) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
    means, stds = demand_moments(reference)
    retailer_bounds: list[tuple[int, int]] = []
    for mean, std, lead_time in zip(means, stds, reference["retailer_lead_times"]):
        lead_periods = int(lead_time) + 1
        lower = max(0, int(math.floor(mean * lead_periods)))
        upper = max(0, int(math.ceil(mean * lead_periods + 3.0 * std * math.sqrt(lead_periods))))
        retailer_bounds.append((lower, upper))
    system_mean = sum(means)
    system_variance = sum(s * s for s in stds)
    cumulative = int(reference["warehouse_lead_time"]) + max(
        int(v) for v in reference["retailer_lead_times"]
    ) + 1
    warehouse_lower = max(0, int(math.floor(system_mean * cumulative)))
    warehouse_upper = max(
        0, int(math.ceil(system_mean * cumulative + 3.0 * math.sqrt(system_variance * cumulative)))
    )
    return (warehouse_lower, warehouse_upper), retailer_bounds, is_symmetric(reference)


# --------------------------------------------------------------------------- simulate / search

def simulate(reference, warehouse_level, retailer_levels, allocation, replications, seed) -> float:
    wi, wp, ri, rp = initial_state(reference)
    mean_cost, _ = ir.one_warehouse_multi_retailer_simulate_policy(
        policy_name="echelon_base_stock",
        params=[float(warehouse_level)] + [float(x) for x in retailer_levels],
        initial_warehouse_inventory=wi,
        initial_warehouse_pipeline=wp,
        initial_retailer_inventory=ri,
        initial_retailer_pipeline=rp,
        periods=int(reference["benchmark_periods"]),
        replications=int(replications),
        seed=int(seed),
        demand_kinds=[str(k) for k in reference["demand_kinds"]],
        demand_param1=[float(x) for x in reference["demand_param1"]],
        demand_param2=[float(x) for x in reference["demand_param2"]],
        holding_cost_warehouse=float(reference["holding_cost_warehouse"]),
        holding_cost_retailers=[float(x) for x in reference["holding_cost_retailers"]],
        penalty_costs_retailers=[float(x) for x in reference["penalty_costs_retailers"]],
        customer_behavior=str(reference["customer_behavior"]),
        emergency_shipment_probability=float(reference["emergency_shipment_probability"]),
        discount_factor=1.0,
        allocation_policy=allocation,
    )
    return float(mean_cost)


def search(reference, allocation, replications, seed):
    (wl, wu), retailer_bounds, symmetric = search_bounds(reference)
    best = None
    if symmetric:
        low, high = retailer_bounds[0]
        candidates = (
            (w, [r] * len(reference["retailer_lead_times"]))
            for w in range(wl, wu + 1)
            for r in range(low, high + 1)
        )
    else:
        grids = [range(l, u + 1) for l, u in retailer_bounds]
        candidates = (
            (w, list(levels))
            for w in range(wl, wu + 1)
            for levels in itertools.product(*grids)
        )
    for warehouse_level, retailer_levels in candidates:
        cost = simulate(reference, warehouse_level, retailer_levels, allocation, replications, seed)
        if best is None or cost < best["mean_cost"]:
            best = {
                "warehouse_base_stock_level": warehouse_level,
                "retailer_base_stock_levels": list(retailer_levels),
                "mean_cost": cost,
            }
    return best


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_names", nargs="+", default=None,
                        help="Subset of Kaynov instance names; default = all 14.")
    parser.add_argument("--search_replications", type=int, default=200)
    parser.add_argument("--eval_replications", type=int, default=1000)
    parser.add_argument("--search_seed", type=int, default=1111)
    parser.add_argument("--eval_seed", type=int, default=2222)
    parser.add_argument("--output_json", default=None)
    parsed = parser.parse_args()

    names = parsed.instance_names or [
        str(r["name"]) for r in ir.one_warehouse_multi_retailer_list_reference_instances()
    ]

    # Env transition/cost cross-check against the repo-native exact DP.
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
    for name in names:
        reference = dict(ir.one_warehouse_multi_retailer_get_reference_instance(name))
        best_prop = search(reference, "proportional", parsed.search_replications, parsed.search_seed)
        best_min = search(reference, "min_shortage", parsed.search_replications, parsed.search_seed)
        eval_prop = simulate(
            reference, best_prop["warehouse_base_stock_level"],
            best_prop["retailer_base_stock_levels"], "proportional",
            parsed.eval_replications, parsed.eval_seed,
        )
        eval_min = simulate(
            reference, best_min["warehouse_base_stock_level"],
            best_min["retailer_base_stock_levels"], "min_shortage",
            parsed.eval_replications, parsed.eval_seed,
        )
        pub_prop = -float(reference["published_proportional_benchmark"]["mean_cost"])
        pub_min = -float(reference["published_min_shortage_benchmark"]["mean_cost"])
        rows.append({
            "instance": name,
            "customer_behavior": reference["customer_behavior"],
            "repo_proportional": eval_prop,
            "published_proportional": pub_prop,
            "gap_pct_proportional": 100.0 * (eval_prop - pub_prop) / pub_prop,
            "repo_min_shortage": eval_min,
            "published_min_shortage": pub_min,
            "gap_pct_min_shortage": 100.0 * (eval_min - pub_min) / pub_min,
            "search_proportional": best_prop,
            "search_min_shortage": best_min,
        })

    payload = {
        "family": "one_warehouse_multi_retailer",
        "source": "Kaynov et al. (2024), IJPE 267, 109088",
        "protocol": {
            "search_replications": parsed.search_replications,
            "eval_replications": parsed.eval_replications,
            "periods": 100,
            "discount_factor": 1.0,
            "initial_state_rule": "mean_filled_pipeline_warm_start",
            "note": "Heuristic-only reproduction; learned soft-tree path is a blocker (see README).",
        },
        "exact_dp_self_consistency": exact_check,
        "rows": rows,
    }

    header = (
        f"{'instance':28s} {'cb':16s} "
        f"{'repoProp':>9s} {'pubProp':>9s} {'gap%':>6s} "
        f"{'repoMin':>9s} {'pubMin':>9s} {'gap%':>6s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['instance']:28s} {row['customer_behavior']:16s} "
            f"{row['repo_proportional']:9.2f} {row['published_proportional']:9.2f} "
            f"{row['gap_pct_proportional']:6.2f} "
            f"{row['repo_min_shortage']:9.2f} {row['published_min_shortage']:9.2f} "
            f"{row['gap_pct_min_shortage']:6.2f}"
        )
    print("\nexact DP self-consistency:", json.dumps(exact_check, indent=1))

    if parsed.output_json:
        Path(parsed.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
