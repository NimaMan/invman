from invman.problems.lost_sales.reference_instances import (
    VANILLA_L4_P4_POISSON5,
    get_benchmark_grid,
    get_reference_instance,
)


def test_lost_sales_grid_contains_four_demand_families():
    grid = get_benchmark_grid("xin2020_extended_lost_sales")
    assert grid["axes"]["lead_time"] == [4, 6, 8, 10]
    assert grid["axes"]["shortage_cost"] == [4, 19]
    assert grid["axes"]["demand_case"] == [
        "poisson",
        "geometric",
        "mmpp2_positive",
        "mmpp2_negative",
    ]
    assert len(grid["instances"]) == 32


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


def test_mmpp2_extension_instance_has_no_published_values():
    instance = get_reference_instance("lit_mmpp2_pos_p4_l4")
    assert instance.params["demand_dist_name"] == "MarkovModulatedPoisson2"
    assert instance.literature_metadata["demand_case"] == "mmpp2_positive"
    assert instance.literature_metadata["reported_values"] == {}
