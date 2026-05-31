"""
Shared orchestration glue for the dual-sourcing benchmark / autoresearch scripts.

OBJECTIVE
    Serve the dual-sourcing CMA-ES policy-search and benchmark scripts after the
    Python-cleanup migration. The Python ``invman.problems.dual_sourcing.*`` and
    ``invman.policies.registry`` trees were deleted; everything now routes through
    the Rust extension ``invman_rust``. This module is the thin, Rust-backed
    replacement for the deleted ``reference_instances.py`` / ``benchmark.py`` /
    ``experiment_spec.py`` helpers, kept inside ``scripts/dual_sourcing/`` (NOT in
    ``invman/``) so the package stays Rust-only while the scripts share one
    implementation instead of duplicating ~150 lines six times.

WHY EACH PIECE EXISTS (every requirement maps to the objective)
    * build_reference_args(name): the dual-sourcing simulator/policy fitness in
      ``invman.rollout_fitness._dual_sourcing_kwargs`` reads the cost/lead-time/
      demand fields straight off ``args``. We populate those from the Rust
      reference instance (``dual_sourcing_get_reference_instance``) so a learned
      soft-tree policy trains on the exact published Figure-9 problem row.
    * get_benchmark_grid / build_grid_instances: source the 6-row Gijsbrechts 2022
      Figure-9 family from ``dual_sourcing_get_experiment_grid`` /
      ``dual_sourcing_expand_experiment_grid`` so the suite iterates the published
      instances with their search/eval protocol and ``published_optimality_gap_pct``.
    * evaluate_default_heuristics(args): the benchmark target. The published metric
      is relative optimality gap, and the strongest heuristic is capped_dual_index.
      We compute single/dual/capped-dual-index + tailored-base-surge costs on a
      fixed demand sample via the Rust search bindings (the heuristic params are
      grid-searched inside Rust, returning the best cost). ``experiment_runner``
      writes an EMPTY ``evaluation.heuristics`` for dual sourcing (its hardcoded
      lookup only knows lost-sales), so the scripts MUST compute heuristics here.
    * bounded_dp_optimal(args): OPT-IN bounded average-cost DP optimum
      (``dual_sourcing_bounded_average_cost_optimal_summary``). It is slow on the
      l_r=4 rows, so it is never on the critical launch path; the best heuristic
      (capped_dual_index, ~0% published gap) is the practical optimal proxy.
    * EXPERIMENT_SPECS / configure_run_args / budgets: the soft-tree policy roster
      and CMA-ES budgets. Dual sourcing's policy backbone is soft_tree ONLY
      (see ``_dual_sourcing_kwargs``); policy search = soft-tree structure (depth,
      temperature, split type, leaf type) over a control adapter (identity or one
      of the dual-index / capped-dual-index / base-surge target adapters).

ALGORITHM (per instance, as used by benchmark_full_suite.py)
    1. Pull the reference instance + protocol from invman_rust.
    2. Draw one fixed demand path (discrete-uniform[demand_low, demand_high]) with
       numpy seed == search_seed, length == search_horizon.
    3. For each heuristic, call its Rust grid-search-from-demands binding -> best
       cost on that path. The minimum over heuristics is the best-heuristic cost.
    4. (opt-in) Solve the bounded average-cost DP for the optimum.
    5. Train each soft-tree policy spec with CMA-ES (run_experiment) and evaluate
       its mean cost over eval_seeds at eval_horizon.
    6. Report learned-policy cost, gap vs best heuristic, gap vs optimal (when
       computed), and the published Figure-9 optimality-gap labels.

All baselines are None-safe: a missing heuristic / optimal yields ``None`` rather
than raising, mirroring the lost-sales suite glue.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np

import invman_rust

from invman.config import get_config
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name

GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME = "gijsbrechts2022_figure9_family"


# --- CMA-ES budgets -----------------------------------------------------------
# Two budgets: a fast screening pass to rank soft-tree structures, and a fuller
# pass for the launched benchmark suite. Mirrors the deleted experiment_spec.py.
COMMON_BUDGET = {
    "screening": {
        "training_episodes": 300,
        "es_population": 8,
        "es_population_sampling": "fixed",
        "horizon": 1000,
        "eval_horizon": 5000,
        "eval_seeds": 2,
        "sigma_init": 3.0,
    },
    "full": {
        "training_episodes": 1500,
        "es_population": 128,
        "es_population_sampling": "categorical",
        "es_population_candidates": [32, 64, 96, 128],
        "es_population_probabilities": [0.05, 0.15, 0.25, 0.55],
        "horizon": 2000,
        "eval_horizon": 10000,
        "eval_seeds": 3,
        "sigma_init": 3.0,
    },
}
DEFAULT_BUDGET = "screening"


# --- Soft-tree policy roster --------------------------------------------------
# Dual sourcing is soft_tree-ONLY. Each spec is a soft-tree structure over a
# control adapter. ``selected`` is the structure promoted by the autoresearch
# screening (axis-aligned constant-leaf small-cap capped-dual-index); the others
# are kept as candidates so the suite reports the full structure comparison.
EXPERIMENT_SPECS = [
    {
        "id": "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
        "label": "Soft tree, axis-constant small-cap capped dual-index",
        "policy_name": "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets",
        "rollout_backend": "rust",
        "status": "selected",
    },
    {
        "id": "soft_tree_capped_dual_index_delta_smallcap_targets",
        "label": "Soft tree, small-cap capped dual-index (oblique linear)",
        "policy_name": "soft_tree_capped_dual_index_delta_smallcap_targets",
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_capped_dual_index_targets",
        "label": "Soft tree, capped dual-index targets (oblique linear)",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="capped_dual_index_targets",
        ),
        "rollout_backend": "rust",
        "status": "candidate",
    },
    {
        "id": "soft_tree_dual_index_targets",
        "label": "Soft tree, dual-index targets (oblique linear)",
        "policy_name": make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="dual_index_targets",
        ),
        "rollout_backend": "rust",
        "status": "candidate",
    },
]


def get_budget_config(budget_name: str) -> dict:
    try:
        return COMMON_BUDGET[budget_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(COMMON_BUDGET))
        raise KeyError(f"Unknown dual-sourcing budget '{budget_name}'. Available: {known}") from exc


# --- reference instance -> args ----------------------------------------------
def _reference_instance(name: str) -> dict:
    inst = invman_rust.dual_sourcing_get_reference_instance(name)
    if inst is None:
        known = ", ".join(invman_rust.dual_sourcing_list_reference_instances.__doc__ or [])
        raise KeyError(f"unknown dual-sourcing reference instance: {name}")
    return dict(inst)


def build_reference_args(name: str = "dual_l4_ce110"):
    """Build an experiment ``args`` populated from a Rust reference instance."""
    inst = _reference_instance(name)
    args = get_config([])
    args.problem = "dual_sourcing"
    args.regular_lead_time = int(inst["regular_lead_time"])
    args.expedited_lead_time = int(inst["expedited_lead_time"])
    args.regular_order_cost = float(inst["regular_order_cost"])
    args.expedited_order_cost = float(inst["expedited_order_cost"])
    args.holding_cost = float(inst["holding_cost"])
    args.shortage_cost = float(inst["shortage_cost"])
    args.regular_max_order_size = int(inst["regular_max_order_size"])
    args.expedited_max_order_size = int(inst["expedited_max_order_size"])
    args.dual_demand_low = int(inst["demand_low"])
    args.dual_demand_high = int(inst["demand_high"])
    args.state_features = "pipeline"
    args.warm_up_periods_ratio = 0.2
    args.track_demand = True
    args.reference_instance = name
    args.seed = 123
    return args


# --- benchmark grid -----------------------------------------------------------
def get_benchmark_grid(name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME) -> dict:
    return dict(invman_rust.dual_sourcing_get_experiment_grid(name))


def build_grid_instances(grid_name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME) -> list[dict]:
    instances = []
    for item in invman_rust.dual_sourcing_expand_experiment_grid(grid_name):
        payload = dict(item)
        payload["params"] = dict(payload["params"])
        payload["search"] = dict(payload["search"])
        payload["evaluation"] = dict(payload["evaluation"])
        payload["literature_metadata"] = dict(payload["literature_metadata"])
        published = payload["literature_metadata"].get("published_optimality_gap_pct")
        if published is not None:
            payload["literature_metadata"]["published_optimality_gap_pct"] = dict(published)
        instances.append(payload)
    return instances


# --- heuristic baselines via the Rust search bindings -------------------------
def _target_upper_bound(args) -> int:
    mean_demand = 0.5 * (int(args.dual_demand_low) + int(args.dual_demand_high))
    upper = int(round((int(args.regular_lead_time) + 2) * mean_demand + 2 * int(args.expedited_max_order_size)))
    return max(int(args.expedited_max_order_size), min(24, upper))


def _fixed_demand_path(args, *, seed: int, horizon: int):
    np.random.seed(int(seed))
    return np.random.randint(int(args.dual_demand_low), int(args.dual_demand_high) + 1, size=int(horizon)).astype(int).tolist()


def evaluate_default_heuristics(args, *, seed: int | None = None, horizon: int | None = None, top_k: int = 3) -> dict:
    """Best cost for each Gijsbrechts heuristic on one fixed demand path.

    Returns ``{heuristic: {"mean_cost": float|None, "params": {...}|None,
    "available": bool, "source": str}}`` for single/dual/capped-dual index and
    tailored-base-surge. Each heuristic's params are grid-searched inside Rust.
    """
    search_seed = int(getattr(args, "seed", 123) if seed is None else seed)
    search_horizon = int(min(int(getattr(args, "horizon", 6000)), 6000) if horizon is None else horizon)
    # The initial pipeline state for the path matches the reference instance.
    inst = _reference_instance(getattr(args, "reference_instance", "dual_l4_ce110"))
    state = [int(v) for v in inst["initial_state"]]
    demands = _fixed_demand_path(args, seed=search_seed, horizon=search_horizon)
    common = dict(
        state=state,
        demands=demands,
        regular_max_order_size=int(args.regular_max_order_size),
        expedited_max_order_size=int(args.expedited_max_order_size),
        regular_order_cost=float(args.regular_order_cost),
        expedited_order_cost=float(args.expedited_order_cost),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        top_k=int(top_k),
        target_upper_bound=_target_upper_bound(args),
    )
    searches = {
        "single_index": (
            invman_rust.dual_sourcing_single_index_search_from_demands,
            lambda best: {"s_e": int(best[0]), "s_r": int(best[1])},
        ),
        "dual_index": (
            invman_rust.dual_sourcing_dual_index_search_from_demands,
            lambda best: {"s_e": int(best[0]), "s_r": int(best[1])},
        ),
        "capped_dual_index": (
            invman_rust.dual_sourcing_capped_dual_index_search_from_demands,
            lambda best: {"s_e": int(best[0]), "s_r": int(best[1]), "cap_r": int(best[2])},
        ),
        "tailored_base_surge": (
            invman_rust.dual_sourcing_tailored_base_surge_search_from_demands,
            lambda best: {"surge_level": int(best[0]), "regular_qty": int(best[1])},
        ),
    }
    results: dict[str, dict] = {}
    for name, (search_fn, params_of) in searches.items():
        try:
            best, _top = search_fn(**common)
            results[name] = {
                "mean_cost": float(best[-1]),
                "params": params_of(best),
                "available": True,
                "source": "rust_search_from_demands",
                "search_seed": search_seed,
                "search_horizon": search_horizon,
            }
        except Exception as exc:  # None-safe: a failed search must not abort the suite
            results[name] = {
                "mean_cost": None,
                "params": None,
                "available": False,
                "source": f"rust_search_failed:{type(exc).__name__}",
            }
    return results


def best_heuristic(heuristics: dict):
    """(name, cost) of the cheapest available heuristic, or (None, None)."""
    available = {k: v for k, v in heuristics.items() if v.get("mean_cost") is not None}
    if not available:
        return None, None
    name = min(available, key=lambda k: available[k]["mean_cost"])
    return name, float(available[name]["mean_cost"])


# --- bounded average-cost DP optimum (OPT-IN; slow on l_r=4) -------------------
def bounded_dp_optimal(args, *, use_grid_bounds: bool = True) -> dict:
    """Bounded average-cost DP optimum for ``args``.

    Slow on l_r in {3,4}; never on the launch critical path. ``use_grid_bounds``
    passes the grid's tight inventory bounds for speed. Returns a None-safe dict.
    """
    kwargs = dict(
        regular_lead_time=int(args.regular_lead_time),
        regular_order_cost=float(args.regular_order_cost),
        expedited_order_cost=float(args.expedited_order_cost),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        regular_max_order_size=int(args.regular_max_order_size),
        expedited_max_order_size=int(args.expedited_max_order_size),
        demand_low=int(args.dual_demand_low),
        demand_high=int(args.dual_demand_high),
    )
    if use_grid_bounds:
        kwargs.update(
            inventory_lower=int(getattr(args, "inventory_lower", -12)),
            inventory_upper=int(getattr(args, "inventory_upper", 24)),
            tolerance=float(getattr(args, "solver_tolerance", 1e-8)),
            max_iterations=int(getattr(args, "max_iterations", 250)),
        )
    try:
        summary = dict(invman_rust.dual_sourcing_bounded_average_cost_optimal_summary(**kwargs))
        return {
            "mean_cost": float(summary["average_cost"]),
            "available": True,
            "source": "rust_bounded_average_cost_dp",
            "iterations": int(summary.get("iterations", 0)),
            "inventory_bounds": list(summary.get("inventory_bounds", [])),
        }
    except Exception as exc:  # None-safe
        return {"mean_cost": None, "available": False, "source": f"rust_dp_failed:{type(exc).__name__}"}


# --- experiment args ----------------------------------------------------------
def _apply_budget(args, budget: dict):
    args.training_method = "cma"
    args.training_episodes = budget["training_episodes"]
    args.es_population = budget["es_population"]
    args.es_population_sampling = budget.get("es_population_sampling", "fixed")
    args.es_population_candidates = budget.get("es_population_candidates")
    args.es_population_probabilities = budget.get("es_population_probabilities")
    args.horizon = budget["horizon"]
    args.eval_horizon = budget["eval_horizon"]
    args.eval_seeds = budget["eval_seeds"]
    args.sigma_init = float(budget["sigma_init"])


def configure_run_args(parsed, spec: dict, root: Path, reference_name: str, *, include_reference_in_experiment_name: bool = True):
    """Build CMA-ES ``args`` for a soft-tree policy spec on a reference instance."""
    budget = get_budget_config(getattr(parsed, "budget", DEFAULT_BUDGET))
    args = build_reference_args(reference_name)
    args.problem = "dual_sourcing"
    args.reference_instance = reference_name
    args.seed = int(getattr(parsed, "seed", 123))
    args.same_seed = bool(getattr(parsed, "same_seed", False))
    args.mp_num_processors = int(getattr(parsed, "mp_num_processors", 4))
    _apply_budget(args, budget)
    if getattr(parsed, "eval_horizon", None) is not None:
        args.eval_horizon = int(parsed.eval_horizon)
    if getattr(parsed, "eval_seeds", None) is not None:
        args.eval_seeds = int(parsed.eval_seeds)
    if getattr(parsed, "training_episodes", None) is not None:
        args.training_episodes = int(parsed.training_episodes)
    if getattr(parsed, "training_horizon", None) is not None:
        args.horizon = int(parsed.training_horizon)
    args.policy_name = spec.get("policy_name", spec["id"])
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


def learned_cost_of(payload: dict) -> float:
    return float(payload["evaluation"]["learned_policy"]["mean_cost"])
