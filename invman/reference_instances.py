from copy import copy
from dataclasses import dataclass

import numpy as np

from invman.config import get_config
from invman.heuristics.lost_sales_heuristics import get_heuristic_policy_cost


@dataclass(frozen=True)
class ReferenceInstance:
    name: str
    params: dict
    expected_costs: dict
    cap_candidates: tuple[int, ...]
    tolerance: float


VANILLA_L4_P4_POISSON5 = ReferenceInstance(
    name="vanilla_l4_p4_poisson5",
    params={
        "problem": "lost_sales",
        "demand_rate": 5.0,
        "lead_time": 4,
        "holding_cost": 1.0,
        "shortage_cost": 4.0,
        "demand_dist_name": "Poisson",
        "max_order_size": 20,
        "horizon": int(1e5),
        "eval_horizon": int(1e5),
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
        "seed": 123,
    },
    expected_costs={
        "optimal": 4.73,
        "myopic1": 5.06,
        "myopic2": 4.82,
        "svbs": 5.83,
        "capped_base_stock": 4.80,
    },
    cap_candidates=(20, 30, 40),
    tolerance=0.12,
)


REFERENCE_INSTANCES = {VANILLA_L4_P4_POISSON5.name: VANILLA_L4_P4_POISSON5}


def get_reference_instance(name=VANILLA_L4_P4_POISSON5.name):
    return REFERENCE_INSTANCES[name]


def build_reference_args(name=VANILLA_L4_P4_POISSON5.name):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance.params.items():
        setattr(args, key, value)
    return args


def _summarize(costs, max_orders, reference_cost):
    mean_cost = float(np.mean(costs))
    return {
        "mean_cost": mean_cost,
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "reference_cost": float(reference_cost),
        "abs_gap": float(abs(mean_cost - reference_cost)),
        "max_order_observed": int(max(max_orders) if max_orders else 0),
        "num_runs": int(len(costs)),
    }


def evaluate_reference_heuristics(name=VANILLA_L4_P4_POISSON5.name, horizon=None, seeds=None, max_order_size=None):
    instance = get_reference_instance(name)
    args = build_reference_args(name)
    if horizon is not None:
        args.horizon = int(horizon)
    if max_order_size is not None:
        args.max_order_size = int(max_order_size)

    if seeds is None:
        seeds = [args.seed]

    results = {}
    for heuristic_name in ("myopic1", "myopic2", "svbs"):
        costs = []
        max_orders = []
        for seed in seeds:
            run_args = copy(args)
            run_args.seed = int(seed)
            env, _, state_action = get_heuristic_policy_cost(run_args, heuristic=heuristic_name)
            costs.append(env.avg_total_cost)
            max_orders.append(max(state_action.values()) if state_action else 0)
        results[heuristic_name] = _summarize(costs, max_orders, instance.expected_costs[heuristic_name])

    return results


def evaluate_cap_sensitivity(name=VANILLA_L4_P4_POISSON5.name, caps=None, seed=None, horizon=None):
    instance = get_reference_instance(name)
    caps = instance.cap_candidates if caps is None else tuple(int(cap) for cap in caps)
    seed = instance.params["seed"] if seed is None else int(seed)
    horizon = instance.params["horizon"] if horizon is None else int(horizon)

    cap_results = {}
    for cap in caps:
        heuristic_results = evaluate_reference_heuristics(
            name=name,
            horizon=horizon,
            seeds=[seed],
            max_order_size=cap,
        )
        cap_results[cap] = {
            heuristic_name: heuristic_summary["mean_cost"]
            for heuristic_name, heuristic_summary in heuristic_results.items()
        }
    return cap_results
