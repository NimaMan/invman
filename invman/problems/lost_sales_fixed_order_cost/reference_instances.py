from copy import deepcopy
from types import SimpleNamespace


REFERENCE_INSTANCES = {
    "starter_l4_p4_k5_poisson5": {
        "name": "starter_l4_p4_k5_poisson5",
        "description": "Starter fixed-order-cost instance aligned with the vanilla calibration case.",
        "params": {
            "problem": "lost_sales_fixed_order_cost",
            "demand_rate": 5.0,
            "lead_time": 4,
            "holding_cost": 1.0,
            "shortage_cost": 4.0,
            "procurement_cost": 0.0,
            "fixed_order_cost": 5.0,
            "demand_dist_name": "Poisson",
            "max_order_size": 50,
            "horizon": 3000,
            "eval_horizon": 50000,
            "track_demand": True,
            "warm_up_periods_ratio": 0.2,
            "seed": 123,
        },
        "search": {
            "position_upper_bound": 45,
            "search_horizon": 3000,
            "search_seed": 123,
            "top_k_s_s_pairs": 12,
            "q_window": 8,
        },
        "evaluation": {
            "eval_horizon": 50000,
            "eval_seeds": 3,
        },
    }
}


def get_reference_instance(name: str = "starter_l4_p4_k5_poisson5"):
    try:
        return deepcopy(REFERENCE_INSTANCES[name])
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(REFERENCE_INSTANCES))
        raise KeyError(f"Unknown fixed-order-cost instance '{name}'. Available: {known}") from exc


def build_reference_args(name: str = "starter_l4_p4_k5_poisson5"):
    instance = get_reference_instance(name)
    return SimpleNamespace(**instance["params"])
