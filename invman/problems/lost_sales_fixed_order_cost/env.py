from invman.problems.lost_sales.env import (
    LostSalesEnv,
    get_model_fitness,
    get_population_fitness,
)
from invman.problems.lost_sales.env import build_env_from_args as build_base_env_from_args


def build_env_from_args(args, horizon=None, track_demand=False):
    if getattr(args, "fixed_order_cost", 0.0) <= 0:
        raise ValueError("fixed_order_cost must be positive for the fixed-order-cost problem.")
    return build_base_env_from_args(args, horizon=horizon, track_demand=track_demand)


__all__ = [
    "LostSalesEnv",
    "build_env_from_args",
    "get_model_fitness",
    "get_population_fitness",
]
