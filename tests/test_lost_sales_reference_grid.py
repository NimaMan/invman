import invman_rust

from scripts.lost_sales.benchmark_full_suite import get_benchmark_grid


def test_lost_sales_grid_contains_four_demand_families():
    grid = get_benchmark_grid("xin2020_extended_lost_sales")
    assert grid["axes"]["lead_time"] == [4, 6, 8, 10]
    assert grid["axes"]["shortage_cost"] == [4, 19]
    assert grid["axes"]["demand_case"] == [
        "poisson",
        "geometric",
        "mmpp2_pos",
        "mmpp2_neg",
    ]
    assert len(grid["instances"]) == 32


def test_canonical_vanilla_instance_kept_as_named_alias():
    instance = invman_rust.lost_sales_reference_costs("vanilla_l4_p4_poisson5")
    assert instance["lead_time"] == 4
    assert instance["shortage_cost"] == 4.0
    assert instance["demand_kind"] == "Poisson"


def test_literature_instance_exposes_reported_values():
    instance = invman_rust.lost_sales_reference_costs("lit_poisson_p4_l4")
    reported = instance["costs"]
    assert reported["optimal"] == 4.73
    assert reported["myopic2"] == 4.82
    assert reported["capped_base_stock"] == 4.80


def test_mmpp2_extension_instance_has_no_published_values():
    instance = invman_rust.lost_sales_reference_costs("lit_mmpp2_pos_p4_l4")
    assert instance["demand_kind"] == "MarkovModulatedPoisson2"
    assert instance["source"] == "computed"
    assert instance["costs"]["optimal"] is None
    assert instance["costs"]["capped_base_stock"] is None
