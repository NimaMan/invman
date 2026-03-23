from __future__ import annotations

from dataclasses import dataclass

from invman.config import get_config


@dataclass(frozen=True)
class MultiEchelonReferenceInstance:
    name: str
    params: dict


MULTI_ECHELON_REFERENCE_INSTANCES = {
    "multi_echelon_setting1": MultiEchelonReferenceInstance(
        name="multi_echelon_setting1",
        params={
            "problem": "multi_echelon",
            "warehouse_lead_time": 2,
            "retailer_lead_time": 2,
            "multi_demand_mean": 5.0,
            "multi_demand_std": 14.0,
            "num_retailers": 10,
            "warehouse_holding_cost": 3.0,
            "retailer_holding_cost": 3.0,
            "warehouse_expedited_cost": 0.0,
            "warehouse_lost_sale_cost": 60.0,
            "expedited_service_prob": 0.8,
            "warehouse_capacity": 100,
            "warehouse_inventory_cap": 1000,
            "retailer_inventory_cap": 100,
            "warehouse_base_stock_levels": [50, 60, 70, 80, 90, 100],
            "retailer_base_stock_levels": [0, 5, 10, 15, 20, 25, 30, 35, 40],
            "horizon": 4000,
            "eval_horizon": 10000,
            "eval_seeds": 3,
            "track_demand": True,
            "warm_up_periods_ratio": 0.2,
            "seed": 123,
        },
    ),
    "multi_echelon_setting2": MultiEchelonReferenceInstance(
        name="multi_echelon_setting2",
        params={
            "problem": "multi_echelon",
            "warehouse_lead_time": 5,
            "retailer_lead_time": 3,
            "multi_demand_mean": 0.0,
            "multi_demand_std": 20.0,
            "num_retailers": 10,
            "warehouse_holding_cost": 3.0,
            "retailer_holding_cost": 3.0,
            "warehouse_expedited_cost": 0.0,
            "warehouse_lost_sale_cost": 60.0,
            "expedited_service_prob": 0.8,
            "warehouse_capacity": 100,
            "warehouse_inventory_cap": 1000,
            "retailer_inventory_cap": 100,
            "warehouse_base_stock_levels": [40, 50, 60, 70, 80, 90, 100],
            "retailer_base_stock_levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "horizon": 4000,
            "eval_horizon": 10000,
            "eval_seeds": 3,
            "track_demand": True,
            "warm_up_periods_ratio": 0.2,
            "seed": 123,
        },
    ),
}


PRIMARY_REFERENCE_INSTANCE = "multi_echelon_setting2"


def get_reference_instance(name: str):
    return MULTI_ECHELON_REFERENCE_INSTANCES[name]


def get_primary_reference_instance():
    return get_reference_instance(PRIMARY_REFERENCE_INSTANCE)


def list_reference_instances():
    return list(MULTI_ECHELON_REFERENCE_INSTANCES)


def build_reference_args(name: str = PRIMARY_REFERENCE_INSTANCE):
    instance = get_reference_instance(name)
    args = get_config([])
    for key, value in instance.params.items():
        setattr(args, key, value)
    return args
