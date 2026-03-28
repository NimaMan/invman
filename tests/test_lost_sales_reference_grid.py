from invman.problems.lost_sales.reference_instances import (
    VANILLA_L4_P4_POISSON5,
    get_benchmark_grid,
    get_reference_instance,
)


def test_lost_sales_grid_contains_twenty_literature_instances():
    grid = get_benchmark_grid("xin2020_extended_lost_sales")
    assert grid["axes"]["lead_time"] == [2, 4, 6, 8, 10]
    assert grid["axes"]["shortage_cost"] == [4, 19]
    assert grid["axes"]["demand_dist_name"] == ["Poisson", "Geometric"]
    assert len(grid["instances"]) == 20


def test_canonical_vanilla_instance_kept_as_named_alias():
    instance = get_reference_instance(VANILLA_L4_P4_POISSON5.name)
    assert instance.params["lead_time"] == 4
    assert instance.params["shortage_cost"] == 4.0
    assert instance.params["demand_dist_name"] == "Poisson"


def test_literature_instance_exposes_reported_values():
    instance = get_reference_instance("lit_poisson_p4_l4")
    reported = instance.literature_metadata["reported_values"]
    assert reported["optimal"] == 4.73
    assert reported["M2"] == 4.82
    assert reported["CappedBS"] == 4.80
