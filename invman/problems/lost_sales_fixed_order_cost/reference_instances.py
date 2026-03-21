from copy import deepcopy
from itertools import product
from types import SimpleNamespace


BASE_INSTANCE_PARAMS = {
    "problem": "lost_sales_fixed_order_cost",
    "demand_rate": 5.0,
    "holding_cost": 1.0,
    "procurement_cost": 0.0,
    "demand_dist_name": "Poisson",
    "max_order_size": 50,
    "horizon": 3000,
    "eval_horizon": 50000,
    "track_demand": True,
    "warm_up_periods_ratio": 0.2,
    "seed": 123,
}

DEFAULT_SEARCH_CONFIG = {
    "position_upper_bound": 45,
    "search_horizon": 3000,
    "search_seed": 123,
    "top_k_s_s_pairs": 12,
    "q_window": 8,
}

DEFAULT_EVALUATION_CONFIG = {
    "eval_horizon": 50000,
    "eval_seeds": 3,
}

BENCHMARK_GRIDS = {
    "literature_subset_poisson_mu5": {
        "name": "literature_subset_poisson_mu5",
        "description": (
            "Subset of the Bijvank-Bhulai-Huh (2015) test bed used as the canonical "
            "fixed-order-cost benchmark for this repo."
        ),
        "shared_params": {
            "demand_rate": 5.0,
            "holding_cost": 1.0,
            "demand_dist_name": "Poisson",
        },
        "axes": {
            "lead_time": [1, 2, 3, 4],
            "shortage_cost": [4.0, 19.0],
            "fixed_order_cost": [5.0, 25.0],
        },
        "search": dict(DEFAULT_SEARCH_CONFIG),
        "evaluation": dict(DEFAULT_EVALUATION_CONFIG),
    }
}


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _build_instance_name(params):
    return (
        "lit_pois_mu"
        f"{_format_number(params['demand_rate'])}_"
        f"l{int(params['lead_time'])}_"
        f"p{_format_number(params['shortage_cost'])}_"
        f"k{_format_number(params['fixed_order_cost'])}"
    )


def get_order_overlap_indicator(params):
    return float(
        2.0
        * params["fixed_order_cost"]
        / (params["demand_rate"] * params["holding_cost"] * (params["lead_time"] ** 2))
    )


def _build_instance_from_params(name, description, params, search, evaluation):
    instance_params = dict(BASE_INSTANCE_PARAMS)
    instance_params.update(params)
    return {
        "name": name,
        "description": description,
        "params": instance_params,
        "search": deepcopy(search),
        "evaluation": deepcopy(evaluation),
        "literature_metadata": {
            "order_overlap_indicator": get_order_overlap_indicator(instance_params),
        },
    }


def build_grid_instances(grid_name: str = "literature_subset_poisson_mu5"):
    try:
        grid = BENCHMARK_GRIDS[grid_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(BENCHMARK_GRIDS))
        raise KeyError(f"Unknown fixed-order-cost grid '{grid_name}'. Available: {known}") from exc

    shared_params = dict(grid["shared_params"])
    axes = grid["axes"]
    axis_names = tuple(axes)

    instances = []
    for axis_values in product(*(axes[axis] for axis in axis_names)):
        instance_params = dict(shared_params)
        for axis_name, axis_value in zip(axis_names, axis_values):
            instance_params[axis_name] = axis_value
        instance_name = _build_instance_name(instance_params)
        description = (
            f"Literature subset instance with Poisson demand mu={_format_number(instance_params['demand_rate'])}, "
            f"L={int(instance_params['lead_time'])}, p={_format_number(instance_params['shortage_cost'])}, "
            f"K={_format_number(instance_params['fixed_order_cost'])}."
        )
        instances.append(
            _build_instance_from_params(
                name=instance_name,
                description=description,
                params=instance_params,
                search=grid["search"],
                evaluation=grid["evaluation"],
            )
        )

    return instances


def get_benchmark_grid(grid_name: str = "literature_subset_poisson_mu5"):
    instances = build_grid_instances(grid_name)
    grid = deepcopy(BENCHMARK_GRIDS[grid_name])
    grid["instances"] = instances
    return grid


REFERENCE_INSTANCES = {
    instance["name"]: instance for grid_name in BENCHMARK_GRIDS for instance in build_grid_instances(grid_name)
}


def list_reference_instances():
    return sorted(REFERENCE_INSTANCES)


def get_reference_instance(name: str = "lit_pois_mu5_l4_p4_k5"):
    try:
        return deepcopy(REFERENCE_INSTANCES[name])
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(list_reference_instances())
        raise KeyError(f"Unknown fixed-order-cost instance '{name}'. Available: {known}") from exc


def build_reference_args(name: str = "lit_pois_mu5_l4_p4_k5"):
    instance = get_reference_instance(name)
    return SimpleNamespace(**instance["params"])
