"""
Config-backed benchmark/experiment-spec/reference surface for the vanilla
lost-sales suite.

The Python-cleanup migration removed invman/problems/lost_sales/{benchmark,
experiment_spec,reference_instances}.py and routed everything through Rust. This
module re-provides the small orchestration surface that
``scripts/lost_sales/benchmark_full_suite.py`` (and the canonical suite) need,
sourcing the benchmark grid and heuristic reference costs from the Rust config
(``problems::lost_sales::reference_costs`` via ``invman_rust``):

  - ``COMMON_BUDGET`` / ``EXPERIMENT_SPECS`` — CMA-ES budget + the policy roster.
  - ``build_reference_args(name)`` — experiment args for a grid instance.
  - ``get_benchmark_grid(name)`` — the 32-instance paper grid.
  - ``benchmark_reference_instance(name)`` — heuristic baselines (myopic1/2/svbs,
    optimal, capped base-stock) read from the config, in the summary shape the
    suite expects.
  - ``configure_run_args`` / ``resolved_protocol_budget`` / ``result_path_for``.
"""

from __future__ import annotations

from pathlib import Path

import invman_rust

from invman.config import get_config
from invman.policy_registry import apply_policy_name

COMMON_BUDGET = {
    "training_episodes_default": 2000,
    "es_population": 64,
    "horizon_default": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
    "save_every": 1000,
}

# Policy roster for the vanilla grid (ids must resolve in invman.policy_registry).
EXPERIMENT_SPECS = [
    {"id": "linear_categorical_quantity_q20", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_sigmoid_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_soft_gated_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_direct_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "linear_hard_gated_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "linear_soft_gated_ordinal_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_ordinal_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "soft_tree_depth1_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
    {"id": "soft_tree_depth2_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
]

_DEFAULT_GRID_NAME = "xin2020_extended_lost_sales"


def _reference(name: str) -> dict:
    ref = invman_rust.lost_sales_reference_costs(name)
    if ref is None:
        raise KeyError(f"unknown lost-sales reference instance: {name}")
    return ref


def _instance_params(ref: dict) -> dict:
    """Problem parameters for a grid instance, matching the old summary shape."""
    params = {
        "problem": "lost_sales",
        "demand_dist_name": ref["demand_kind"],
        "demand_rate": ref["demand_rate"],
        "lead_time": int(ref["lead_time"]),
        "holding_cost": ref["holding_cost"],
        "shortage_cost": ref["shortage_cost"],
        "max_order_size": 20,
        "horizon": COMMON_BUDGET["horizon_default"],
        "eval_horizon": COMMON_BUDGET["eval_horizon"],
        "eval_seeds": COMMON_BUDGET["eval_seeds"],
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
        "seed": 123,
    }
    if ref["demand_kind"] == "MarkovModulatedPoisson2":
        params.update(
            demand_lambda_low=ref["demand_lambda_low"],
            demand_lambda_high=ref["demand_lambda_high"],
            demand_p00=ref["demand_p00"],
            demand_p11=ref["demand_p11"],
        )
    return params


def build_reference_args(name: str):
    """Experiment args for a grid instance (problem/demand/cost params set)."""
    ref = _reference(name)
    args = get_config([])
    args.problem = "lost_sales"
    args.demand_dist_name = ref["demand_kind"]
    args.demand_rate = ref["demand_rate"]
    args.lead_time = int(ref["lead_time"])
    args.holding_cost = ref["holding_cost"]
    args.shortage_cost = ref["shortage_cost"]
    args.max_order_size = 20
    args.track_demand = True
    args.warm_up_periods_ratio = 0.2
    if ref["demand_kind"] == "MarkovModulatedPoisson2":
        args.demand_lambda_low = ref["demand_lambda_low"]
        args.demand_lambda_high = ref["demand_lambda_high"]
        args.demand_p00 = ref["demand_p00"]
        args.demand_p11 = ref["demand_p11"]
    return args


def get_benchmark_grid(grid_name: str = _DEFAULT_GRID_NAME) -> dict:
    """The 32-instance vanilla paper grid (excludes the canonical vanilla alias)."""
    instances = []
    for name in invman_rust.lost_sales_reference_instance_names():
        if name == "vanilla_l4_p4_poisson5":
            continue
        ref = _reference(name)
        instances.append(
            {
                "name": name,
                "description": (
                    f"Lost-sales grid instance: {ref['demand_kind']} demand (mean 5), "
                    f"lead time {ref['lead_time']}, shortage cost {int(ref['shortage_cost'])}, "
                    f"holding cost 1."
                ),
                "params": _instance_params(ref),
                "literature_metadata": {
                    "benchmark_family": "Xin2020TechnicalModels",
                    "reference_cost_source": ref["source"],
                    "reported_values": dict(ref["costs"]),
                },
            }
        )
    return {
        "name": grid_name,
        "grid_name": grid_name,
        "description": (
            "Vanilla lost-sales paper grid: {Poisson, Geometric, MMPP2+, MMPP2-} demand x "
            "shortage cost {4, 19} x lead time {4, 6, 8, 10}, mean demand 5, holding cost 1."
        ),
        "axes": {
            "lead_time": [4, 6, 8, 10],
            "shortage_cost": [4, 19],
            "demand_case": ["poisson", "geometric", "mmpp2_pos", "mmpp2_neg"],
        },
        "num_instances": len(instances),
        "instances": instances,
    }


def _cost_summary(value):
    return {
        "mean_cost": None if value is None else float(value),
        "available": value is not None,
        "source": "reference_config",
    }


def benchmark_reference_instance(name: str, *, eval_horizon=None, eval_seeds=None, **_ignored) -> dict:
    """Heuristic reference summary for an instance, read from the config.

    `eval_horizon`/`eval_seeds` are accepted for call-compatibility with the old
    simulation-based function but ignored: the costs are precomputed.
    """
    ref = _reference(name)
    costs = ref["costs"]
    return {
        "reference_instance": name,
        "evaluation": {
            "myopic1": _cost_summary(costs["myopic1"]),
            "myopic2": _cost_summary(costs["myopic2"]),
            "svbs": _cost_summary(costs["svbs"]),
        },
        "optimal_reference": _cost_summary(costs["optimal"]),
        "capped_base_stock_reference": _cost_summary(costs["capped_base_stock"]),
        "reference_cost_source": ref["source"],
    }


def _resolve_budget(parsed) -> dict:
    training_episodes = (
        int(parsed.training_episodes)
        if getattr(parsed, "training_episodes", None) is not None
        else COMMON_BUDGET["training_episodes_default"]
    )
    horizon = (
        int(parsed.training_horizon)
        if getattr(parsed, "training_horizon", None) is not None
        else COMMON_BUDGET["horizon_default"]
    )
    return {"training_episodes": training_episodes, "horizon": horizon}


def resolved_protocol_budget(parsed) -> dict:
    budget = _resolve_budget(parsed)
    return {
        "training_episodes_default": budget["training_episodes"],
        "horizon_default": budget["horizon"],
    }


def configure_run_args(parsed, spec, root: Path, reference_name: str, *, include_reference_in_experiment_name: bool = True):
    args = build_reference_args(reference_name)
    budget = _resolve_budget(parsed)
    args.problem = "lost_sales"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes"]
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = budget["horizon"]
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
    args.save_every = COMMON_BUDGET["save_every"]
    args.max_order_size = 20
    args.policy_name = spec["id"]
    apply_policy_name(args)
    args.rollout_backend = spec["rollout_backend"]
    args.results_dir = str(root / "results")
    args.log_dir = str(root / "logs")
    args.trained_models_dir = str(root / "models")
    if include_reference_in_experiment_name:
        args.experiment_name = f"{parsed.run_tag}_{reference_name}_{spec['id']}"
    else:
        args.experiment_name = f"{parsed.run_tag}_{spec['id']}"
    return args


def result_path_for(args) -> Path:
    return Path(args.results_dir) / f"{args.experiment_name}.json"
