from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from invman.config import get_config

try:
    import invman_rust
except ImportError:  # pragma: no cover - fallback when the Rust extension is unavailable
    invman_rust = None


@dataclass(frozen=True)
class PublishedBenchmarkReference:
    source: str
    url: str
    benchmark_policies: tuple[str, ...]
    published_values: dict


@dataclass(frozen=True)
class DualSourcingReferenceInstance:
    name: str
    params: dict
    expected_ranking: tuple[str, ...]
    benchmark_policies: tuple[str, ...]
    literature_values: dict


DUAL_SOURCING_BENCHMARK_REFERENCE = PublishedBenchmarkReference(
    source="Gijsbrechts et al. (2022), Section 6.2 / Figure 9",
    url="https://doi.org/10.1287/msom.2021.1064",
    benchmark_policies=(
        "optimal_dp",
        "single_index",
        "dual_index",
        "capped_dual_index",
        "tailored_base_surge",
        "lp_adp",
        "a3c",
    ),
    published_values={
        "a3c_optimality_gap_pct_upper": 2.0,
        "strongest_heuristic_in_gijsbrechts2022": "capped_dual_index",
        "published_metric": "optimality_gap_pct",
        "instance_family_source": "Veeraraghavan and Scheller-Wolf (2008)",
        "heuristic_family_sources": [
            "Veeraraghavan and Scheller-Wolf (2008)",
            "Sheopuri et al. (2010)",
        ],
        "notes": (
            "Figure 9 prints per-instance optimality-gap labels for the main heuristics and A3C, "
            "but the paper does not provide a per-instance table of absolute heuristic or optimal costs."
        ),
    },
)


_HEURISTIC_TIE_BREAK = (
    "capped_dual_index",
    "tailored_base_surge",
    "dual_index",
    "single_index",
)

GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME = "gijsbrechts2022_figure9_family"

DEFAULT_SEARCH_CONFIG = {
    "inventory_lower": -12,
    "inventory_upper": 24,
    "tolerance": 1e-8,
    "max_iterations": 250,
    "search_seed": 123,
    "search_horizon": 6000,
}

DEFAULT_EVALUATION_CONFIG = {
    "eval_horizon": 20000,
    "eval_seeds": 3,
}


def _expected_ranking_from_gaps(published_optimality_gap_pct: dict[str, float]) -> tuple[str, ...]:
    return tuple(
        policy
        for policy, _ in sorted(
            (
                (policy, float(published_optimality_gap_pct[policy]))
                for policy in _HEURISTIC_TIE_BREAK
            ),
            key=lambda item: (item[1], _HEURISTIC_TIE_BREAK.index(item[0])),
        )
    )


DUAL_SOURCING_PUBLISHED_FIGURE9_GAPS = {
    "dual_l2_ce105": {
        "capped_dual_index": 0.00,
        "dual_index": 0.11,
        "single_index": 0.56,
        "tailored_base_surge": 0.06,
        "a3c": 0.52,
    },
    "dual_l2_ce110": {
        "capped_dual_index": 0.03,
        "dual_index": 0.18,
        "single_index": 1.03,
        "tailored_base_surge": 0.99,
        "a3c": 0.80,
    },
    "dual_l3_ce105": {
        "capped_dual_index": 0.00,
        "dual_index": 0.27,
        "single_index": 0.98,
        "tailored_base_surge": 0.01,
        "a3c": 0.82,
    },
    "dual_l3_ce110": {
        "capped_dual_index": 0.06,
        "dual_index": 0.36,
        "single_index": 2.11,
        "tailored_base_surge": 0.71,
        "a3c": 0.51,
    },
    "dual_l4_ce105": {
        "capped_dual_index": 0.00,
        "dual_index": 0.36,
        "single_index": 1.43,
        "tailored_base_surge": 0.00,
        "a3c": 1.85,
    },
    "dual_l4_ce110": {
        "capped_dual_index": 0.11,
        "dual_index": 0.49,
        "single_index": 2.44,
        "tailored_base_surge": 0.58,
        "a3c": 1.33,
    },
}


def _build_instance(name: str, regular_lead_time: int, expedited_order_cost: float):
    published_optimality_gap_pct = DUAL_SOURCING_PUBLISHED_FIGURE9_GAPS.get(name, {})
    return DualSourcingReferenceInstance(
        name=name,
        params={
            "problem": "dual_sourcing",
            "regular_lead_time": int(regular_lead_time),
            "expedited_lead_time": 0,
            "regular_order_cost": 100.0,
            "expedited_order_cost": float(expedited_order_cost),
            "holding_cost": 5.0,
            "shortage_cost": 495.0,
            "regular_max_order_size": 12,
            "expedited_max_order_size": 12,
            "dual_demand_low": 0,
            "dual_demand_high": 4,
            "horizon": 6000,
            "eval_horizon": 20000,
            "eval_seeds": 3,
            "track_demand": True,
            "warm_up_periods_ratio": 0.2,
            "state_features": "pipeline",
            "seed": 123,
        },
        expected_ranking=_expected_ranking_from_gaps(published_optimality_gap_pct),
        benchmark_policies=DUAL_SOURCING_BENCHMARK_REFERENCE.benchmark_policies,
        literature_values={
            "source": DUAL_SOURCING_BENCHMARK_REFERENCE.source,
            "instance_family_source": DUAL_SOURCING_BENCHMARK_REFERENCE.published_values["instance_family_source"],
            "heuristic_family_sources": tuple(
                DUAL_SOURCING_BENCHMARK_REFERENCE.published_values["heuristic_family_sources"]
            ),
            "a3c_optimality_gap_pct_upper": 2.0,
            "best_reported_heuristic_family": "capped_dual_index",
            "literature_verified": True,
            "literature_verification_metric": "published_relative_optimality_gap_pct",
            "published_optimality_gap_pct": {key: float(value) for key, value in published_optimality_gap_pct.items()},
            "has_exact_published_cost": False,
        },
    )


DUAL_SOURCING_REFERENCE_INSTANCES = {
    "dual_l2_ce105": _build_instance("dual_l2_ce105", regular_lead_time=2, expedited_order_cost=105.0),
    "dual_l2_ce110": _build_instance("dual_l2_ce110", regular_lead_time=2, expedited_order_cost=110.0),
    "dual_l3_ce105": _build_instance("dual_l3_ce105", regular_lead_time=3, expedited_order_cost=105.0),
    "dual_l3_ce110": _build_instance("dual_l3_ce110", regular_lead_time=3, expedited_order_cost=110.0),
    "dual_l4_ce105": _build_instance("dual_l4_ce105", regular_lead_time=4, expedited_order_cost=105.0),
    "dual_l4_ce110": _build_instance("dual_l4_ce110", regular_lead_time=4, expedited_order_cost=110.0),
}


PRIMARY_REFERENCE_INSTANCE = "dual_l4_ce110"


BENCHMARK_GRIDS = {
    GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME: {
        "name": GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME,
        "description": "Six small-scale dual-sourcing benchmark rows from Gijsbrechts et al. (2022), Section 6.2 / Figure 9.",
        "source": DUAL_SOURCING_BENCHMARK_REFERENCE.source,
        "url": DUAL_SOURCING_BENCHMARK_REFERENCE.url,
        "reference_instance_names": list(DUAL_SOURCING_REFERENCE_INSTANCES),
        "regular_lead_times": [2, 3, 4],
        "expedited_order_costs": [105.0, 110.0],
        "regular_order_cost": 100.0,
        "holding_cost": 5.0,
        "shortage_cost": 495.0,
        "demand_low": 0,
        "demand_high": 4,
        "regular_max_order_size": 12,
        "expedited_max_order_size": 12,
        "horizon": 6000,
        "eval_horizon": 20000,
        "eval_seeds": 3,
        "search_seed": 123,
        "inventory_lower": -12,
        "inventory_upper": 24,
        "solver_tolerance": 1e-8,
        "max_iterations": 250,
        "warm_up_periods_ratio": 0.2,
        "state_features": "pipeline",
        "notes": (
            "This grid matches the six published Figure 9 problem rows. "
            "The literature verification target is the published relative optimality gaps, "
            "not unpublished absolute costs."
        ),
    }
}


def get_reference_instance(name: str):
    return DUAL_SOURCING_REFERENCE_INSTANCES[name]


def get_primary_reference_instance():
    return get_reference_instance(PRIMARY_REFERENCE_INSTANCE)


def list_reference_instances():
    return list(DUAL_SOURCING_REFERENCE_INSTANCES)


def get_benchmark_reference():
    return DUAL_SOURCING_BENCHMARK_REFERENCE


def _build_grid_instances_from_rust(grid_name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME):
    if invman_rust is None or not hasattr(invman_rust, "dual_sourcing_expand_experiment_grid"):
        return None
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


def _build_grid_instances_manual(grid_name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME):
    try:
        grid = BENCHMARK_GRIDS[grid_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(BENCHMARK_GRIDS))
        raise KeyError(f"Unknown dual-sourcing grid '{grid_name}'. Available: {known}") from exc

    instances = []
    for reference_name in grid["reference_instance_names"]:
        reference = get_reference_instance(reference_name)
        params = deepcopy(reference.params)
        search = deepcopy(DEFAULT_SEARCH_CONFIG)
        evaluation = deepcopy(DEFAULT_EVALUATION_CONFIG)
        literature_metadata = {
            "source": reference.literature_values["source"],
            "url": DUAL_SOURCING_BENCHMARK_REFERENCE.url,
            "literature_verified": True,
            "literature_verification_metric": "published_relative_optimality_gap_pct",
            "benchmark_family": "Gijsbrechts2022Figure9DualSourcing",
            "benchmark_policies": list(reference.benchmark_policies),
            "published_optimality_gap_pct": deepcopy(
                reference.literature_values["published_optimality_gap_pct"]
            ),
            "notes": (
                "Gijs Figure 9 dual-sourcing benchmark row with literature verification defined "
                "against the published relative optimality gaps."
            ),
        }
        instances.append(
            {
                "name": reference.name,
                "description": (
                    "Gijsbrechts Figure 9 dual-sourcing benchmark row with "
                    f"l_r={params['regular_lead_time']}, c_e={params['expedited_order_cost']:.0f}, "
                    "c_r=100, h=5, b=495, and demand U{0,1,2,3,4}."
                ),
                "reference_instance_name": reference.name,
                "source": reference.literature_values["source"],
                "url": DUAL_SOURCING_BENCHMARK_REFERENCE.url,
                "params": params,
                "search": search,
                "evaluation": evaluation,
                "literature_metadata": literature_metadata,
            }
        )
    return instances


def get_benchmark_grid(name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME):
    if invman_rust is not None and hasattr(invman_rust, "dual_sourcing_get_experiment_grid"):
        return dict(invman_rust.dual_sourcing_get_experiment_grid(name))
    try:
        return deepcopy(BENCHMARK_GRIDS[name])
    except KeyError as exc:  # pragma: no cover - defensive programming
        known = ", ".join(sorted(BENCHMARK_GRIDS))
        raise KeyError(f"Unknown dual-sourcing grid '{name}'. Available: {known}") from exc


def build_grid_instances(grid_name: str = GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME):
    rust_instances = _build_grid_instances_from_rust(grid_name)
    if rust_instances is not None:
        return rust_instances
    return _build_grid_instances_manual(grid_name)


def build_reference_args(name: str = PRIMARY_REFERENCE_INSTANCE):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance.params.items():
        setattr(args, key, value)
    setattr(args, "reference_instance", name)
    return args
