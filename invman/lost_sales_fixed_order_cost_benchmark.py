"""
Config-backed benchmark/experiment-spec/reference surface for the fixed-order-cost
lost-sales suite (post-migration).

The Python-cleanup migration removed
invman/problems/lost_sales_fixed_order_cost/{benchmark,experiment_spec,
reference_instances}.py that the suite runner imported. This module re-provides
the orchestration surface that
``scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py`` needs.

The fixed-cost heuristic family ((s,S), (s,nQ), modified (s,S,q)) has not yet
been ported to Rust, so ``benchmark_reference_instance`` returns EMPTY heuristic
baselines: the suite trains learned policies via the Rust fitness path but does
not (re)compute heuristic comparisons. The already-completed instances keep
their baselines in their existing summaries; port (s,S,q) to Rust later to
restore baselines for new instances.

The 64-instance grid is generated from its axes (demand x lead time x shortage
cost x setup cost K); instance naming matches the original
``lit_{token}_mu5_l{L}_p{p}_k{K}`` so ``--reuse_existing`` resumes cleanly.
"""

from __future__ import annotations

from pathlib import Path

from invman.config import get_config
from invman.policy_registry import apply_policy_name

COMMON_BUDGET = {
    "training_episodes": 2000,
    "es_population": 64,
    "horizon": 2000,
    "dynamic_horizon": False,
    "min_dynamic_horizon": 2000,
    "max_dynamic_horizon": 2000,
    "eval_horizon": int(1e6),
    "eval_seeds": 10,
    "sigma_init": 5.0,
}

EXPERIMENT_SPECS = [
    {"id": "linear_soft_gated_direct_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_direct_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "linear_soft_gated_ordinal_quantity", "rollout_backend": "rust", "status": "trusted"},
    {"id": "nn_soft_gated_ordinal_quantity_h8_selu", "rollout_backend": "rust", "status": "provisional"},
    {"id": "soft_tree_depth1_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
    {"id": "soft_tree_depth2_linear_leaf", "rollout_backend": "rust", "status": "trusted"},
]

FULL_GRID_NAME = "lost_sales_style_full_grid_mu5"

# token -> (demand_dist_name, MMPP2 demand params or None)
_DEMAND_CASES = {
    "pois": ("Poisson", None),
    "geom": ("Geometric", None),
    "mmpp2_pos": ("MarkovModulatedPoisson2", dict(low=3.0, high=7.0, p00=0.9, p11=0.9)),
    "mmpp2_neg": ("MarkovModulatedPoisson2", dict(low=3.0, high=7.0, p00=0.1, p11=0.1)),
}
_LEAD_TIMES = [4, 6, 8, 10]
_SHORTAGE_COSTS = [4, 19]
_SETUP_COSTS = [5, 25]


def _grid_specs():
    """Yield (name, token, lead_time, shortage_cost, setup_cost) for all 64 instances."""
    for token in _DEMAND_CASES:
        for lead_time in _LEAD_TIMES:
            for shortage_cost in _SHORTAGE_COSTS:
                for setup_cost in _SETUP_COSTS:
                    name = f"lit_{token}_mu5_l{lead_time}_p{shortage_cost}_k{setup_cost}"
                    yield name, token, lead_time, shortage_cost, setup_cost


_INSTANCES = {
    name: {"token": token, "lead_time": L, "shortage_cost": p, "setup_cost": k}
    for name, token, L, p, k in _grid_specs()
}


def _instance(name: str) -> dict:
    if name not in _INSTANCES:
        raise KeyError(f"unknown fixed-cost reference instance: {name}")
    return _INSTANCES[name]


def _params(name: str) -> dict:
    inst = _instance(name)
    demand_name, mmpp2 = _DEMAND_CASES[inst["token"]]
    params = {
        "problem": "lost_sales_fixed_order_cost",
        "demand_dist_name": demand_name,
        "demand_rate": 5.0,
        "lead_time": inst["lead_time"],
        "shortage_cost": float(inst["shortage_cost"]),
        "fixed_order_cost": float(inst["setup_cost"]),
        "holding_cost": 1.0,
        "procurement_cost": 0.0,
        "max_order_size": 20,
        "track_demand": True,
        "warm_up_periods_ratio": 0.2,
    }
    if mmpp2 is not None:
        params.update(
            demand_lambda_low=mmpp2["low"],
            demand_lambda_high=mmpp2["high"],
            demand_p00=mmpp2["p00"],
            demand_p11=mmpp2["p11"],
        )
    return params


def build_reference_args(name: str):
    """Experiment args for a fixed-cost grid instance."""
    args = get_config([])
    for key, value in _params(name).items():
        setattr(args, key, value)
    # canonical state representation used by the original runs
    args.state_normalizer = "quantity_scale"
    args.state_scale = 20.0
    return args


def get_reference_instance(name: str | None = None) -> dict:
    """Instance descriptor. No published-optimal anchor is encoded, so the suite
    reports the optimal reference as unavailable."""
    if name is None:
        name = next(iter(_INSTANCES))
    _instance(name)
    return {"name": name, "params": _params(name)}


def get_benchmark_grid(grid_name: str = FULL_GRID_NAME) -> dict:
    instances = []
    for name, token, L, p, k in _grid_specs():
        demand_name, _ = _DEMAND_CASES[token]
        instances.append(
            {
                "name": name,
                "description": (
                    f"Fixed-cost lost-sales grid instance: {demand_name} demand (mean 5), "
                    f"lead time {L}, shortage cost {p}, setup cost {k}, holding cost 1."
                ),
                "params": _params(name),
                "literature_metadata": {
                    "benchmark_family": "repo_fixed_cost_lost_sales_full_grid",
                    "demand_case": token,
                },
            }
        )
    return {
        "name": grid_name,
        "grid_name": grid_name,
        "description": (
            "Fixed-order-cost lost-sales full grid: {Poisson, Geometric, MMPP2+, MMPP2-} demand x "
            "shortage cost {4, 19} x lead time {4, 6, 8, 10} x setup cost {5, 25}, mean demand 5, "
            "holding cost 1."
        ),
        "axes": {
            "lead_time": _LEAD_TIMES,
            "shortage_cost": _SHORTAGE_COSTS,
            "fixed_order_cost": _SETUP_COSTS,
            "demand_case": list(_DEMAND_CASES),
        },
        "num_instances": len(instances),
        "instances": instances,
    }


def _empty_cost():
    return {"mean_cost": None, "available": False, "source": "not_ported_to_rust"}


def benchmark_reference_instance(name: str, *, eval_horizon=None, eval_seeds=None, **_ignored) -> dict:
    """Empty heuristic summary: the fixed-cost (s,S,q) family is not yet in Rust.

    Returns the summary shape the suite expects, with no heuristic baselines, so
    learned policies still train and are recorded (comparative gaps are None).
    """
    _instance(name)  # validate
    return {
        "reference_instance": name,
        "evaluation": {},
        "optimal_reference": _empty_cost(),
        "capped_base_stock_reference": _empty_cost(),
        "note": "fixed-cost (s,S,q) heuristics not yet ported to Rust; baselines omitted",
    }


def _resolve_budget(parsed) -> dict:
    return {
        "training_episodes": int(
            parsed.training_episodes
            if getattr(parsed, "training_episodes", None) is not None
            else COMMON_BUDGET["training_episodes"]
        ),
        "horizon": int(
            parsed.training_horizon
            if getattr(parsed, "training_horizon", None) is not None
            else COMMON_BUDGET["horizon"]
        ),
    }


def resolved_protocol_budget(parsed) -> dict:
    return _resolve_budget(parsed)


def configure_run_args(parsed, spec, root: Path, reference_name: str, *, include_reference_in_experiment_name: bool = True):
    args = build_reference_args(reference_name)
    budget = _resolve_budget(parsed)
    args.problem = "lost_sales_fixed_order_cost"
    args.reference_instance = reference_name
    args.seed = parsed.seed
    args.same_seed = parsed.same_seed
    args.mp_num_processors = parsed.mp_num_processors
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes"]
    args.es_population = COMMON_BUDGET["es_population"]
    args.horizon = budget["horizon"]
    args.dynamic_horizon = COMMON_BUDGET["dynamic_horizon"]
    args.min_dynamic_horizon = COMMON_BUDGET["min_dynamic_horizon"]
    args.max_dynamic_horizon = COMMON_BUDGET["max_dynamic_horizon"]
    args.eval_horizon = parsed.eval_horizon
    args.eval_seeds = parsed.eval_seeds
    args.sigma_init = COMMON_BUDGET["sigma_init"]
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
