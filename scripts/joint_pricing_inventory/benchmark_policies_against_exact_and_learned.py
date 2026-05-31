# Benchmark joint_pricing_inventory policies on two complementary instances.
#
# ALGORITHM / WHAT THIS SCRIPT DOES
# ---------------------------------
# This script runs the feasible, rigorous benchmark for the joint_pricing_inventory problem
# WITHOUT rebuilding the Rust extension and WITHOUT retraining. It uses only the installed
# `invman_rust` bindings plus already-trained soft-tree parameters stored under
# outputs/joint_pricing_inventory/.
#
# It reports two complementary comparisons:
#
# 1) EXACT-DP-ANCHORED GAPS (verifier instance: 5 periods, discrete price-dependent demand).
#    Here an exact finite-horizon DP optimum exists. We report the optimal discounted cost and the
#    two repo heuristics' exact discounted costs (all computed by the Rust DP), and the profit
#    optimality gap of each heuristic. This instance also carries an INDEPENDENT analytical anchor:
#    its T=1 reduction is the price-setting newsvendor (critical fractile), verified in
#    rust/.../verification/tests.rs.
#
# 2) LEARNED-VS-HEURISTIC ON THE PRIMARY INSTANCE (18 periods, Poisson price-dependent demand).
#    No exact optimum exists here (large/continuing state), so we cannot report an optimality gap.
#    We instead compare the trained soft-tree policy (loaded from stored flat params) against the
#    two repo heuristics on FRESH held-out seeds, reporting mean discounted cost (= -profit) and the
#    learned policy's profit improvement over the best heuristic.
#
# Profit convention: the env returns discounted COST = -profit, so a more negative cost is better
# (higher profit). All "profit" numbers below are -cost.
#
# Usage:
#   python scripts/joint_pricing_inventory/benchmark_policies_against_exact_and_learned.py \
#       --replications 4096 --seed 777000 \
#       --trained_json outputs/joint_pricing_inventory/tree_primary_d2_linear_b8_s123_e120_eval2048.json
#
# NOTE on the learned-policy-on-verifier-instance gap (NOT computed here): producing a TRUE
# optimality gap for a learned policy requires training a soft-tree ON the verifier instance and then
# comparing to its exact DP optimum. That needs the (currently missing) Python SoftTreePolicy class
# / a CMA-ES training pass; see the package README "Remaining steps". This script deliberately avoids
# retraining to respect the no-rebuild / no-contention constraints.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import invman_rust


def exact_dp_anchored_table() -> dict:
    summary = dict(invman_rust.joint_pricing_inventory_exact_dp_summary())
    opt = float(summary["optimal_discounted_cost"])
    static_cost = float(summary["static_discounted_cost"])
    inv_cost = float(summary["inventory_sensitive_discounted_cost"])
    opt_profit = -opt
    return {
        "optimal": {
            "cost": opt,
            "profit": -opt,
            "first_action": tuple(summary["optimal_first_action"]),
        },
        "static_price_base_stock": {
            "cost": static_cost,
            "profit": -static_cost,
            "first_action": tuple(summary["static_first_action"]),
            "profit_optimality_gap_pct": (opt_profit - (-static_cost)) / abs(opt_profit) * 100.0,
        },
        "inventory_sensitive_base_stock": {
            "cost": inv_cost,
            "profit": -inv_cost,
            "first_action": tuple(summary["inventory_sensitive_first_action"]),
            "profit_optimality_gap_pct": (opt_profit - (-inv_cost)) / abs(opt_profit) * 100.0,
        },
    }


def _soft_tree_rollout_kwargs(reference: dict, cfg: dict, flat_params, seed: int) -> dict:
    return dict(
        flat_params=[float(v) for v in flat_params],
        input_dim=7,
        depth=int(cfg["depth"]),
        min_values=[0, 0],
        max_values=[int(reference["max_order_quantity"]), len(reference["price_levels"]) - 1],
        action_mode="vector_quantity",
        inventory_level=int(reference["initial_inventory_level"]),
        periods=int(reference["periods"]),
        demand_kind=str(reference["demand_distribution_kind"]),
        price_levels=[float(v) for v in reference["price_levels"]],
        demand_means=[float(v) for v in reference["price_demand_means"]],
        procurement_cost_per_unit=float(reference["procurement_cost_per_unit"]),
        holding_cost_per_unit=float(reference["holding_cost_per_unit"]),
        stockout_cost_per_unit=float(reference["stockout_cost_per_unit"]),
        salvage_value_per_unit=float(reference["salvage_value_per_unit"]),
        max_order_quantity=int(reference["max_order_quantity"]),
        seed=int(seed),
        discount_factor=0.99,
        temperature=float(cfg["temperature"]),
        split_type=str(cfg["split_type"]),
        leaf_type=str(cfg["leaf_type"]),
        allowed_values=None,
    )


def _simulate_heuristic(reference: dict, name: str, params, replications: int, seed: int) -> dict:
    summary = dict(
        invman_rust.joint_pricing_inventory_simulate_policy(
            policy_name=name,
            params=[float(v) for v in params],
            inventory_level=int(reference["initial_inventory_level"]),
            periods=int(reference["periods"]),
            replications=int(replications),
            seed=int(seed),
            demand_kind=str(reference["demand_distribution_kind"]),
            price_levels=[float(v) for v in reference["price_levels"]],
            demand_means=[float(v) for v in reference["price_demand_means"]],
            procurement_cost_per_unit=float(reference["procurement_cost_per_unit"]),
            holding_cost_per_unit=float(reference["holding_cost_per_unit"]),
            stockout_cost_per_unit=float(reference["stockout_cost_per_unit"]),
            max_order_quantity=int(reference["max_order_quantity"]),
            discount_factor=0.99,
            salvage_value_per_unit=float(reference["salvage_value_per_unit"]),
        )
    )
    return {"mean_cost": float(summary["mean_discounted_cost"]), "cost_std": float(summary["std_discounted_cost"])}


def primary_learned_vs_heuristic(trained_json: Path, replications: int, seed: int) -> dict:
    reference = dict(invman_rust.joint_pricing_inventory_primary_reference_instance())
    payload = json.loads(Path(trained_json).read_text())
    cfg = payload["tree_config"]
    flat = np.asarray(payload["trained_flat_params"], dtype=np.float32)

    costs = [
        invman_rust.joint_pricing_inventory_soft_tree_rollout(
            **_soft_tree_rollout_kwargs(reference, cfg, flat, seed + i)
        )
        for i in range(replications)
    ]
    costs = np.asarray(costs, dtype=np.float64)
    soft_tree = {"mean_cost": float(costs.mean()), "cost_std": float(costs.std())}

    static = _simulate_heuristic(
        reference,
        "static_price_base_stock",
        [reference["benchmark_static_order_up_to"], reference["benchmark_static_price_index"]],
        replications,
        seed,
    )
    inv = _simulate_heuristic(
        reference,
        "inventory_sensitive_base_stock",
        [
            reference["benchmark_inventory_sensitive_order_up_to"],
            reference["benchmark_markdown_threshold"],
            reference["benchmark_high_price_index"],
            reference["benchmark_low_price_index"],
        ],
        replications,
        seed,
    )
    best_heuristic_cost = min(static["mean_cost"], inv["mean_cost"])
    return {
        "instance": reference["name"],
        "replications": int(replications),
        "eval_seed_base": int(seed),
        "tree_config": cfg,
        "soft_tree": soft_tree,
        "static_price_base_stock": static,
        "inventory_sensitive_base_stock": inv,
        "soft_tree_profit_improvement_over_best_heuristic_pct": (
            (best_heuristic_cost - soft_tree["mean_cost"]) / abs(best_heuristic_cost) * 100.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument(
        "--trained_json",
        default="outputs/joint_pricing_inventory/tree_primary_d2_linear_b8_s123_e120_eval2048.json",
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    result = {
        "exact_dp_anchored_verifier_instance": exact_dp_anchored_table(),
        "primary_instance_learned_vs_heuristic": primary_learned_vs_heuristic(
            Path(args.trained_json), args.replications, args.seed
        ),
    }
    text = json.dumps(result, indent=2, default=str)
    if args.output_json:
        Path(args.output_json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
