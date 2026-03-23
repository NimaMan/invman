from __future__ import annotations

from dataclasses import dataclass

from invman.config import get_config


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
    source="Gijsbrechts et al. (2022), Section 6.2 / Figure 9; Veeraraghavan and Scheller-Wolf (2008); Sheopuri et al. (2010)",
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
        "notes": (
            "The paper reports relative performance and ranking on the six settings, "
            "not a per-instance table of exact heuristic or optimal costs."
        ),
    },
)


def _build_instance(name: str, regular_lead_time: int, expedited_order_cost: float):
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
        expected_ranking=("capped_dual_index", "tailored_base_surge", "dual_index", "single_index"),
        benchmark_policies=DUAL_SOURCING_BENCHMARK_REFERENCE.benchmark_policies,
        literature_values={
            "source": DUAL_SOURCING_BENCHMARK_REFERENCE.source,
            "a3c_optimality_gap_pct_upper": 2.0,
            "best_reported_heuristic_family": "capped_dual_index",
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


def get_reference_instance(name: str):
    return DUAL_SOURCING_REFERENCE_INSTANCES[name]


def get_primary_reference_instance():
    return get_reference_instance(PRIMARY_REFERENCE_INSTANCE)


def list_reference_instances():
    return list(DUAL_SOURCING_REFERENCE_INSTANCES)


def get_benchmark_reference():
    return DUAL_SOURCING_BENCHMARK_REFERENCE


def build_reference_args(name: str = PRIMARY_REFERENCE_INSTANCE):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance.params.items():
        setattr(args, key, value)
    return args
