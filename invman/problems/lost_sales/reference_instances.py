from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np

from invman.config import get_config
from invman.problems.lost_sales.heuristics import get_heuristic_policy_cost
from invman.problems.lost_sales.problem_info import problem_info


@dataclass(frozen=True)
class ReferenceInstance:
    name: str
    params: dict
    expected_costs: dict
    cap_candidates: tuple[int, ...]
    tolerance: float
    description: str
    literature_metadata: dict
    benchmark_policy_families: tuple[str, ...]
    heuristic_max_order_size: int = 200


def _make_reference_instance(
    *,
    name: str,
    demand_dist_name: str,
    shortage_cost: int,
    lead_time: int,
) -> ReferenceInstance:
    problem_key = f"{demand_dist_name}_demand_shortage_cost_{shortage_cost}"
    literature_values = problem_info[problem_key][lead_time]
    params = {
        "problem": "lost_sales",
        "demand_rate": 5.0,
        "lead_time": lead_time,
        "holding_cost": 1.0,
        "shortage_cost": float(shortage_cost),
        "demand_dist_name": demand_dist_name,
        "max_order_size": 20,
        "horizon": 2000,
        "eval_horizon": int(1e6),
        "eval_seeds": 10,
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
        "seed": 123,
        "state_normalizer": "quantity_scale",
        "state_scale": 20.0,
    }
    description = (
        f"Literature-aligned lost-sales instance with {demand_dist_name} demand, "
        f"mean demand 5, lead time {lead_time}, shortage cost {shortage_cost}, and holding cost 1."
    )
    literature_metadata = {
        "benchmark_family": "Xin2020TechnicalModels",
        "parent_reference": "Zipkin2008OldSystems",
        "problem_info_key": problem_key,
        "reported_values": dict(literature_values),
        "notes": (
            "This repository uses the 20-instance vanilla lost-sales family from Xin (2020), "
            "which extends the classic Zipkin (2008) set to larger lead times."
        ),
    }
    return ReferenceInstance(
        name=name,
        params=params,
        expected_costs={
            "optimal": literature_values.get("optimal"),
            "myopic1": literature_values.get("M1"),
            "myopic2": literature_values.get("M2"),
            "svbs": literature_values.get("SVBS"),
            "capped_base_stock": literature_values.get("CappedBS"),
        },
        cap_candidates=(8, 20),
        tolerance=0.15,
        description=description,
        literature_metadata=literature_metadata,
        benchmark_policy_families=(
            "myopic1",
            "myopic2",
            "svbs",
            "capped_base_stock",
            "linear_categorical_quantity_q8",
            "linear_categorical_quantity_q20",
            "nn_categorical_quantity_q8",
            "nn_categorical_quantity_q20",
            "soft_tree_depth2_linear_leaf_q8",
        ),
        heuristic_max_order_size=200,
    )


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
        "horizon": 2000,
        "eval_horizon": int(1e6),
        "eval_seeds": 10,
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
        "seed": 123,
        "state_normalizer": "quantity_scale",
        "state_scale": 20.0,
    },
    expected_costs={
        "optimal": 4.73,
        "myopic1": 5.06,
        "myopic2": 4.82,
        "svbs": 5.83,
        "capped_base_stock": 4.80,
    },
    cap_candidates=(8, 20),
    tolerance=0.12,
    description=(
        "Canonical vanilla lost-sales benchmark with Poisson demand, mean demand 5, lead time 4, "
        "and shortage cost 4."
    ),
    literature_metadata={
        "benchmark_family": "Xin2020TechnicalModels",
        "parent_reference": "Zipkin2008OldSystems",
        "reported_values": {
            "optimal": 4.73,
            "myopic1": 5.06,
            "myopic2": 4.82,
            "svbs": 5.83,
            "capped_base_stock": 4.80,
        },
        "notes": "This is the canonical L=4, p=4, Poisson(5) instance used throughout the repository.",
    },
    benchmark_policy_families=(
        "myopic1",
        "myopic2",
        "svbs",
        "capped_base_stock",
        "linear_categorical_quantity_q8",
        "linear_categorical_quantity_q20",
        "nn_categorical_quantity_q8",
        "nn_categorical_quantity_q20",
        "soft_tree_depth2_linear_leaf_q8",
    ),
    heuristic_max_order_size=200,
)


_LITERATURE_GRID_INSTANCES = tuple(
    _make_reference_instance(
        name=f"lit_{demand_dist_name.lower()}_p{shortage_cost}_l{lead_time}",
        demand_dist_name=demand_dist_name,
        shortage_cost=shortage_cost,
        lead_time=lead_time,
    )
    for demand_dist_name in ("Poisson", "Geometric")
    for shortage_cost in (4, 19)
    for lead_time in (2, 4, 6, 8, 10)
)


REFERENCE_INSTANCES = {instance.name: instance for instance in _LITERATURE_GRID_INSTANCES}
REFERENCE_INSTANCES[VANILLA_L4_P4_POISSON5.name] = VANILLA_L4_P4_POISSON5


BENCHMARK_GRIDS = {
    "xin2020_extended_lost_sales": {
        "name": "xin2020_extended_lost_sales",
        "description": (
            "Twenty literature-aligned lost-sales instances from Xin (2020): "
            "lead times {2,4,6,8,10}, shortage costs {4,19}, and demand distributions "
            "{Poisson, Geometric}, all with mean demand 5 and holding cost 1."
        ),
        "axes": {
            "lead_time": [2, 4, 6, 8, 10],
            "shortage_cost": [4, 19],
            "demand_dist_name": ["Poisson", "Geometric"],
            "demand_rate": [5.0],
        },
        "instances": [
            {
                "name": instance.name,
                "description": instance.description,
                "params": instance.params,
                "literature_metadata": instance.literature_metadata,
            }
            for instance in _LITERATURE_GRID_INSTANCES
        ],
    }
}


def get_reference_instance(name=VANILLA_L4_P4_POISSON5.name):
    return REFERENCE_INSTANCES[name]


def get_benchmark_grid(name: str = "xin2020_extended_lost_sales"):
    return BENCHMARK_GRIDS[name]


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
        "reference_cost": None if reference_cost is None or (isinstance(reference_cost, float) and np.isnan(reference_cost)) else float(reference_cost),
        "abs_gap": None
        if reference_cost is None or (isinstance(reference_cost, float) and np.isnan(reference_cost))
        else float(abs(mean_cost - reference_cost)),
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
    else:
        args.max_order_size = int(instance.heuristic_max_order_size)

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
