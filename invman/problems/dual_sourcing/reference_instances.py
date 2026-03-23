from __future__ import annotations

from dataclasses import dataclass

from invman.config import get_config


@dataclass(frozen=True)
class DualSourcingReferenceInstance:
    name: str
    params: dict
    expected_ranking: tuple[str, ...]


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


def build_reference_args(name: str = PRIMARY_REFERENCE_INSTANCE):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance.params.items():
        setattr(args, key, value)
    return args
