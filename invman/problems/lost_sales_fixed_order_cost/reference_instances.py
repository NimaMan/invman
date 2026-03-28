from copy import deepcopy
from itertools import product

from invman.config import get_config


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

CANONICAL_REFERENCE_NAME = "lit_pois_mu5_l4_p4_k5"
PUBLISHED_VALIDATION_REFERENCE_NAME = "bijvank2015_table1_l2_p14_k5"

BENCHMARK_ANCHORS = {
    PUBLISHED_VALIDATION_REFERENCE_NAME: {
        "benchmark_type": "published_validation",
        "description": (
            "Published validation example from Bijvank, Bhulai, and Huh (2015), Table 1."
        ),
        "literature_source": {
            "citation_key": "Bijvank2015ParametricPolicies",
            "notes": (
                "Example with R=1, L=2, h=1, p=14, K=5, Poisson demand with mean 5."
            ),
        },
        "published_optimal_reference": {
            "available": True,
            "mean_cost": 11.46,
        },
        "published_heuristic_references": {
            "s_s": {
                "params": {"s": 17, "S": 23},
                "mean_cost": 11.62,
            },
            "s_nq": {
                "params": {"s": 17, "q": 7},
                "mean_cost": 11.56,
            },
            "modified_s_s_q": {
                "params": {"s": 17, "S": 23, "q": 7},
                "mean_cost": 11.50,
            },
        },
    },
    CANONICAL_REFERENCE_NAME: {
        "benchmark_type": "repo_canonical",
        "description": (
            "Canonical fixed-order-cost benchmark instance aligned with the Bijvank-Bhulai-Huh "
            "(2015) test-bed subset used in this repo."
        ),
        "evaluation_protocol": {
            "long_run_eval_horizon": int(1e6),
            "warm_up_periods_ratio": 0.2,
            "eval_seeds": 10,
            "training_episodes": 5000,
            "training_horizon": 2000,
            "es_population": 50,
            "sigma_init": 5.0,
        },
        "heuristic_anchors_1m": {
            "s_s": {
                "params": {"s": 21, "S": 27},
                "mean_cost": 9.371451375000001,
            },
            "s_nq": {
                "params": {"s": 22, "q": 8},
                "mean_cost": 9.180962249999999,
            },
            "modified_s_s_q": {
                "params": {"s": 22, "S": 30, "q": 8},
                "mean_cost": 9.17435925,
            },
        },
        "paper_like_policy_suite_1m": {
            "linear_categorical_quantity": 10.272988999999999,
            "linear_gated_ordinal_quantity": 8.768776124999999,
            "nn_categorical_quantity": 10.272988999999999,
            "nn_gated_ordinal_quantity": 8.732815500000001,
            "soft_tree_depth2_linear_leaf": 8.774177125,
            "soft_tree_depth1_linear_leaf": 8.778456875,
        },
        "policy_approximator_anchors": {
            "linear_categorical_quantity": {
                "policy_type": "linear",
                "policy_head": "categorical_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 10.272988999999999,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "linear_gated_ordinal_quantity": {
                "policy_type": "linear",
                "policy_head": "gated_ordinal_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 8.768776124999999,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "nn_categorical_quantity": {
                "policy_type": "nn",
                "policy_head": "categorical_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 10.272988999999999,
                "training_episodes": 5000,
                "verification_status": "needs_verification",
                "note": (
                    "Current paper-like benchmark matched the linear categorical baseline exactly; "
                    "treat as provisional until re-verified."
                ),
            },
            "nn_gated_ordinal_quantity": {
                "policy_type": "nn",
                "policy_head": "gated_ordinal_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 8.732815500000001,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "soft_tree_depth2_linear_leaf": {
                "policy_type": "soft_tree",
                "policy_head": "tree_linear_leaf_quantity",
                "tree_depth": 2,
                "eval_horizon": int(1e6),
                "mean_cost": 8.774177125,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "soft_tree_depth1_linear_leaf": {
                "policy_type": "soft_tree",
                "policy_head": "tree_linear_leaf_quantity",
                "tree_depth": 1,
                "eval_horizon": int(1e6),
                "mean_cost": 8.778456875,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
        },
        "legacy_exploration_anchors": {
            "historical_linear_50k": 10.423691666666667,
            "historical_nn_gated_ordinal_50k": 9.516358333333333,
            "transferred_tree_depth2_1m": 8.810086666666669,
            "current_best_tree_depth1_1m": 8.765759583333333,
        },
    }
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

MANUAL_REFERENCE_INSTANCES = {
    PUBLISHED_VALIDATION_REFERENCE_NAME: {
        "name": PUBLISHED_VALIDATION_REFERENCE_NAME,
        "description": (
            "Published validation example from Bijvank, Bhulai, and Huh (2015), Table 1."
        ),
        "params": {
            "lead_time": 2,
            "shortage_cost": 14.0,
            "fixed_order_cost": 5.0,
        },
        "search": {
            "position_upper_bound": 31,
            "search_horizon": 10000,
            "search_seed": 123,
            "top_k_s_s_pairs": 12,
            "q_window": 8,
        },
        "evaluation": {
            "eval_horizon": 200000,
            "eval_seeds": 10,
        },
        "literature_metadata": {
            "source": "Bijvank2015ParametricPolicies",
            "reported_review_period": 1.0,
            "reported_lead_time": 2,
            "reported_demand_dist_name": "Poisson",
            "reported_demand_mean_per_review_period": 5.0,
        },
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
    instance = {
        "name": name,
        "description": description,
        "params": instance_params,
        "search": deepcopy(search),
        "evaluation": deepcopy(evaluation),
        "literature_metadata": {
            "order_overlap_indicator": get_order_overlap_indicator(instance_params),
        },
    }
    if name in BENCHMARK_ANCHORS:
        instance["benchmark_anchors"] = deepcopy(BENCHMARK_ANCHORS[name])
    return instance


def _build_manual_reference_instances():
    instances = []
    for instance in MANUAL_REFERENCE_INSTANCES.values():
        instance_params = dict(BASE_INSTANCE_PARAMS)
        instance_params.update(instance["params"])
        payload = {
            "name": instance["name"],
            "description": instance["description"],
            "params": instance_params,
            "search": deepcopy(instance["search"]),
            "evaluation": deepcopy(instance["evaluation"]),
            "literature_metadata": deepcopy(instance["literature_metadata"]),
        }
        payload["literature_metadata"]["order_overlap_indicator"] = get_order_overlap_indicator(instance_params)
        if payload["name"] in BENCHMARK_ANCHORS:
            payload["benchmark_anchors"] = deepcopy(BENCHMARK_ANCHORS[payload["name"]])
        instances.append(payload)
    return instances


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
    instance["name"]: instance
    for instance in [
        *[instance for grid_name in BENCHMARK_GRIDS for instance in build_grid_instances(grid_name)],
        *_build_manual_reference_instances(),
    ]
}


def list_reference_instances():
    return sorted(REFERENCE_INSTANCES)


def get_reference_instance(name: str = CANONICAL_REFERENCE_NAME):
    try:
        return deepcopy(REFERENCE_INSTANCES[name])
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(list_reference_instances())
        raise KeyError(f"Unknown fixed-order-cost instance '{name}'. Available: {known}") from exc


def build_reference_args(name: str = CANONICAL_REFERENCE_NAME):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance["params"].items():
        setattr(args, key, value)
    return args
