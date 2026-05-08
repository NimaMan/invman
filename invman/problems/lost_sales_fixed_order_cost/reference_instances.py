from copy import deepcopy
from itertools import product
from math import ceil, sqrt

from invman.config import get_config
from invman.problems.lost_sales.demand import (
    MMPP2_NEGATIVE_MEAN5,
    MMPP2_POSITIVE_MEAN5,
    build_demand_config,
)

try:
    import invman_rust
except ImportError:  # pragma: no cover - fallback when the Rust extension is unavailable
    invman_rust = None


_MMPP2_POSITIVE_CONFIG = build_demand_config(**MMPP2_POSITIVE_MEAN5)
_MMPP2_NEGATIVE_CONFIG = build_demand_config(**MMPP2_NEGATIVE_MEAN5)


BASE_INSTANCE_PARAMS = {
    "problem": "lost_sales_fixed_order_cost",
    "demand_rate": 5.0,
    "holding_cost": 1.0,
    "procurement_cost": 0.0,
    "demand_dist_name": "Poisson",
    "max_order_size": 50,
    "horizon": 3000,
    "eval_horizon": 50000,
    "warm_up_periods_ratio": 0.2,
    "seed": 42,
    "state_normalizer": "quantity_scale",
    "state_scale": 50.0,
}

DEFAULT_SEARCH_CONFIG = {
    "position_upper_bound": 45,
    "search_horizon": 3000,
    "search_seed": 42,
    "top_k_s_s_pairs": 12,
    "q_window": 8,
}

DEFAULT_EVALUATION_CONFIG = {
    "eval_horizon": 50000,
    "eval_seeds": 3,
}

CANONICAL_REFERENCE_NAME = "lit_pois_mu5_l4_p4_k5"
PUBLISHED_VALIDATION_REFERENCE_NAME = "bijvank2015_table1_l2_p14_k5"
CORRELATED_POSITIVE_REFERENCE_NAME = "corr_mmpp2_pos_mu5_l4_p4_k5"
CORRELATED_NEGATIVE_REFERENCE_NAME = "corr_mmpp2_neg_mu5_l4_p4_k5"
FULL_GRID_NAME = "lost_sales_style_full_grid_mu5"

DEMAND_CASE_SPECS = {
    "poisson": {
        "display_name": "Poisson",
        "name_token": "pois",
        "params": {
            "demand_dist_name": "Poisson",
            "demand_rate": 5.0,
        },
        "notes": "Poisson demand with mean 5.",
    },
    "geometric": {
        "display_name": "Geometric",
        "name_token": "geom",
        "params": {
            "demand_dist_name": "Geometric",
            "demand_rate": 5.0,
        },
        "notes": "Geometric demand with mean 5.",
    },
    "mmpp2_positive": {
        "display_name": "MMPP2 positive",
        "name_token": "mmpp2_pos",
        "params": dict(MMPP2_POSITIVE_MEAN5),
        "notes": (
            "Two-state Markov-modulated Poisson demand with positive lag-1 autocorrelation and "
            "stationary mean 5."
        ),
    },
    "mmpp2_negative": {
        "display_name": "MMPP2 negative",
        "name_token": "mmpp2_neg",
        "params": dict(MMPP2_NEGATIVE_MEAN5),
        "notes": (
            "Two-state Markov-modulated Poisson demand with negative lag-1 autocorrelation and "
            "stationary mean 5."
        ),
    },
}

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
            "linear_soft_gated_ordinal_quantity": 8.768776124999999,
            "nn_categorical_quantity": 10.272988999999999,
            "nn_soft_gated_ordinal_quantity": 8.732815500000001,
            "soft_tree_depth2_linear_leaf": 8.77689475,
            "soft_tree_depth1_linear_leaf": 8.781105125,
        },
        "policy_approximator_anchors": {
            "linear_categorical_quantity": {
                "policy_name": "linear_categorical_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 10.272988999999999,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "linear_soft_gated_ordinal_quantity": {
                "policy_name": "linear_soft_gated_ordinal_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 8.768776124999999,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "nn_categorical_quantity": {
                "policy_name": "nn_categorical_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 10.272988999999999,
                "training_episodes": 5000,
                "verification_status": "needs_verification",
                "note": (
                    "Current paper-like benchmark matched the linear categorical baseline exactly; "
                    "treat as provisional until re-verified."
                ),
            },
            "nn_soft_gated_ordinal_quantity": {
                "policy_name": "nn_soft_gated_ordinal_quantity",
                "eval_horizon": int(1e6),
                "mean_cost": 8.732815500000001,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "soft_tree_depth2_linear_leaf": {
                "policy_name": "soft_tree_depth2_linear_leaf",
                "tree_depth": 2,
                "eval_horizon": int(1e6),
                "mean_cost": 8.77689475,
                "training_episodes": 5000,
                "verification_status": "trusted",
            },
            "soft_tree_depth1_linear_leaf": {
                "policy_name": "soft_tree_depth1_linear_leaf",
                "tree_depth": 1,
                "eval_horizon": int(1e6),
                "mean_cost": 8.781105125,
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
    },
    FULL_GRID_NAME: {
        "name": FULL_GRID_NAME,
        "description": (
            "Fixed-cost lost-sales paper grid aligned with the vanilla lost-sales benchmark shape: "
            "lead times {4,6,8,10}, shortage costs {4,19}, fixed costs {5,25}, and demand "
            "families {Poisson, Geometric, MMPP2 positive, MMPP2 negative}, all with mean demand 5."
        ),
        "axes": {
            "demand_case": ["poisson", "geometric", "mmpp2_positive", "mmpp2_negative"],
            "lead_time": [4, 6, 8, 10],
            "shortage_cost": [4.0, 19.0],
            "fixed_order_cost": [5.0, 25.0],
        },
    },
    "correlated_mmpp2_mu5_l4_p4_k5": {
        "name": "correlated_mmpp2_mu5_l4_p4_k5",
        "description": (
            "Two mean-preserving correlated-demand fixed-cost robustness cases based on the canonical "
            "L=4, p=4, K=5 instance."
        ),
        "instances": [
            {
                "name": CORRELATED_POSITIVE_REFERENCE_NAME,
                "description": (
                    "Canonical fixed-cost instance with mean-preserving positively correlated "
                    "Markov-modulated Poisson demand."
                ),
                "params": {
                    "lead_time": 4,
                    "shortage_cost": 4.0,
                    "fixed_order_cost": 5.0,
                    **MMPP2_POSITIVE_MEAN5,
                },
                "search": {
                    **DEFAULT_SEARCH_CONFIG,
                    "position_upper_bound": 50,
                },
                "evaluation": dict(DEFAULT_EVALUATION_CONFIG),
                "literature_metadata": {
                    "benchmark_family": "repo_correlated_demand_extension",
                    "notes": (
                        "Two-state Markov-modulated Poisson demand with lambda_low=3, lambda_high=7, "
                        "p00=p11=0.9, stationary mean 5, and lag-1 demand autocorrelation "
                        f"{_MMPP2_POSITIVE_CONFIG.lag_k_autocorrelation(1):.6f}."
                    ),
                },
            },
            {
                "name": CORRELATED_NEGATIVE_REFERENCE_NAME,
                "description": (
                    "Canonical fixed-cost instance with mean-preserving negatively correlated "
                    "Markov-modulated Poisson demand."
                ),
                "params": {
                    "lead_time": 4,
                    "shortage_cost": 4.0,
                    "fixed_order_cost": 5.0,
                    **MMPP2_NEGATIVE_MEAN5,
                },
                "search": {
                    **DEFAULT_SEARCH_CONFIG,
                    "position_upper_bound": 50,
                },
                "evaluation": dict(DEFAULT_EVALUATION_CONFIG),
                "literature_metadata": {
                    "benchmark_family": "repo_correlated_demand_extension",
                    "notes": (
                        "Two-state Markov-modulated Poisson demand with lambda_low=3, lambda_high=7, "
                        "p00=p11=0.1, stationary mean 5, and lag-1 demand autocorrelation "
                        f"{_MMPP2_NEGATIVE_CONFIG.lag_k_autocorrelation(1):.6f}."
                    ),
                },
            },
        ],
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
            "search_seed": 42,
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
    demand_case = _infer_demand_case_key(params)
    token = DEMAND_CASE_SPECS[demand_case]["name_token"]
    return (
        f"lit_{token}_mu"
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


def _infer_demand_case_key(params):
    demand_name = params["demand_dist_name"]
    if demand_name == "Poisson":
        return "poisson"
    if demand_name == "Geometric":
        return "geometric"
    if demand_name == "MarkovModulatedPoisson2":
        p00 = float(params.get("demand_p00", 0.0))
        p11 = float(params.get("demand_p11", 0.0))
        if (
            abs(p00 - float(MMPP2_POSITIVE_MEAN5["demand_p00"])) < 1e-12
            and abs(p11 - float(MMPP2_POSITIVE_MEAN5["demand_p11"])) < 1e-12
        ):
            return "mmpp2_positive"
        if (
            abs(p00 - float(MMPP2_NEGATIVE_MEAN5["demand_p00"])) < 1e-12
            and abs(p11 - float(MMPP2_NEGATIVE_MEAN5["demand_p11"])) < 1e-12
        ):
            return "mmpp2_negative"
    raise ValueError(f"Unsupported fixed-cost demand configuration: {params}")


def _default_position_upper_bound_for_params(params):
    config = build_demand_config(
        demand_dist_name=params["demand_dist_name"],
        demand_rate=params["demand_rate"],
        demand_lambda_low=params.get("demand_lambda_low", MMPP2_POSITIVE_MEAN5["demand_lambda_low"]),
        demand_lambda_high=params.get("demand_lambda_high", MMPP2_POSITIVE_MEAN5["demand_lambda_high"]),
        demand_p00=params.get("demand_p00", MMPP2_POSITIVE_MEAN5["demand_p00"]),
        demand_p11=params.get("demand_p11", MMPP2_POSITIVE_MEAN5["demand_p11"]),
    )
    protection_mean = (int(params["lead_time"]) + 1) * float(params["demand_rate"])
    protection_std = sqrt(float(config.cumulative_variance(int(params["lead_time"]) + 1)))
    return max(1, int(ceil(protection_mean + 4.0 * protection_std)))


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


def _build_full_grid_instances():
    instances = []
    for demand_case_key, lead_time, shortage_cost, fixed_order_cost in product(
        BENCHMARK_GRIDS[FULL_GRID_NAME]["axes"]["demand_case"],
        BENCHMARK_GRIDS[FULL_GRID_NAME]["axes"]["lead_time"],
        BENCHMARK_GRIDS[FULL_GRID_NAME]["axes"]["shortage_cost"],
        BENCHMARK_GRIDS[FULL_GRID_NAME]["axes"]["fixed_order_cost"],
    ):
        demand_case = DEMAND_CASE_SPECS[demand_case_key]
        instance_params = {
            "lead_time": lead_time,
            "shortage_cost": shortage_cost,
            "fixed_order_cost": fixed_order_cost,
            **demand_case["params"],
        }
        instance_name = _build_instance_name(instance_params)
        description = (
            f"Fixed-cost lost-sales full-grid instance with {demand_case['display_name']} demand, "
            f"mean demand {_format_number(instance_params['demand_rate'])}, L={lead_time}, "
            f"p={_format_number(shortage_cost)}, and K={_format_number(fixed_order_cost)}."
        )
        instances.append(
            {
                "name": instance_name,
                "description": description,
                "params": instance_params,
                "search": {
                    **DEFAULT_SEARCH_CONFIG,
                    "position_upper_bound": _default_position_upper_bound_for_params(
                        {**BASE_INSTANCE_PARAMS, **instance_params}
                    ),
                },
                "evaluation": dict(DEFAULT_EVALUATION_CONFIG),
                "literature_metadata": {
                    "benchmark_family": "repo_fixed_cost_lost_sales_full_grid",
                    "parent_problem_family": "Bijvank2015ParametricPolicies",
                    "demand_case": demand_case_key,
                    "demand_case_display_name": demand_case["display_name"],
                    "notes": (
                        "Fixed-cost extension of the lost-sales paper grid with the extra setup-cost "
                        "axis K in {5,25}; Poisson and Geometric slices align with the literature-style "
                        "single-item benchmark surface, while the two MMPP2 slices are repo extensions "
                        "used to match the four-demand benchmark shape of the vanilla lost-sales suite. "
                        + demand_case["notes"]
                    ),
                },
            }
        )
    return instances


def _build_full_grid_instances_from_rust():
    if invman_rust is None:
        return None
    allowed_lead_times = set(BENCHMARK_GRIDS[FULL_GRID_NAME]["axes"]["lead_time"])
    instances = []
    for item in invman_rust.lost_sales_fixed_order_cost_expand_experiment_grid(FULL_GRID_NAME):
        payload = dict(item)
        payload["params"] = dict(payload["params"])
        if int(payload["params"]["lead_time"]) not in allowed_lead_times:
            continue
        payload["search"] = dict(payload["search"])
        payload["evaluation"] = dict(payload["evaluation"])
        payload["literature_metadata"] = dict(payload["literature_metadata"])
        payload["literature_metadata"]["order_overlap_indicator"] = get_order_overlap_indicator(payload["params"])
        if payload["name"] in BENCHMARK_ANCHORS:
            payload["benchmark_anchors"] = deepcopy(BENCHMARK_ANCHORS[payload["name"]])
        instances.append(payload)
    return instances


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


def build_grid_instances(grid_name: str = FULL_GRID_NAME):
    try:
        grid = BENCHMARK_GRIDS[grid_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(BENCHMARK_GRIDS))
        raise KeyError(f"Unknown fixed-order-cost grid '{grid_name}'. Available: {known}") from exc

    if grid_name == FULL_GRID_NAME:
        rust_instances = _build_full_grid_instances_from_rust()
        if rust_instances is not None:
            return rust_instances
        instances = []
        for instance in _build_full_grid_instances():
            instances.append(
                _build_instance_from_params(
                    name=instance["name"],
                    description=instance["description"],
                    params=instance["params"],
                    search=instance["search"],
                    evaluation=instance["evaluation"],
                )
            )
            instances[-1]["literature_metadata"].update(deepcopy(instance.get("literature_metadata", {})))
        return instances

    explicit_instances = grid.get("instances")
    if explicit_instances is not None:
        instances = []
        for instance in explicit_instances:
            instances.append(
                _build_instance_from_params(
                    name=instance["name"],
                    description=instance["description"],
                    params=instance["params"],
                    search=instance["search"],
                    evaluation=instance["evaluation"],
                )
            )
            instances[-1]["literature_metadata"].update(deepcopy(instance.get("literature_metadata", {})))
        return instances

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


def get_benchmark_grid(grid_name: str = FULL_GRID_NAME):
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
