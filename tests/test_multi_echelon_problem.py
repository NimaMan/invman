import pytest

invman_rust = pytest.importorskip("invman_rust")


def test_multi_echelon_reference_instances_match_gijs_metadata():
    benchmark = invman_rust.multi_echelon_benchmark_reference()
    instances = invman_rust.multi_echelon_list_reference_instances()
    primary = invman_rust.multi_echelon_primary_reference_instance()
    van_roy = invman_rust.multi_echelon_van_roy_case_study()

    assert "constant_base_stock" in benchmark["benchmark_policies"]
    assert len(instances) == 5
    assert instances[0]["name"] == "van_roy1997_simple_problem"
    assert instances[0]["published_constant_base_stock_mean_cost"] == pytest.approx(
        51.7
    )
    assert instances[1]["name"] == "van_roy1997_case_study1"
    assert instances[2]["name"] == "van_roy1997_case_study2"
    assert instances[3]["name"] == "gijsbrechts2022_setting1"
    assert instances[4]["name"] == "gijsbrechts2022_setting2"
    assert primary["name"] == "gijsbrechts2022_setting2"
    assert instances[1]["published_a3c_savings_pct"] == pytest.approx(8.95)
    assert instances[2]["published_a3c_savings_pct"] == pytest.approx(12.09)
    assert instances[1]["literature_metadata"]["literature_reference_present"] is True
    assert (
        instances[1]["literature_metadata"]["implementation_literature_verified"]
        is False
    )
    assert (
        instances[1]["literature_metadata"]["literature_verification_metric"]
        == "published_relative_a3c_savings_vs_constant_base_stock_pct"
    )
    assert (
        instances[1]["literature_metadata"]["repo_algorithm_literature_verified"]
        is False
    )
    assert instances[4]["literature_metadata"]["literature_reference_present"] is False
    assert instances[4]["inventory_dynamics_mode"] == "gijs_2022"
    assert instances[4]["demand_mean"] == 0.0
    assert instances[4]["published_a3c_savings_pct"] is None
    assert instances[4]["benchmark_warehouse_levels"] == [50, 60, 70, 80, 90, 100]
    assert instances[4]["benchmark_retailer_levels"] == [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ]
    assert van_roy["published_constant_base_stock_mean_cost"] == pytest.approx(1302.0)
    assert van_roy["published_constant_base_stock_levels"] == [330, 23]


def test_multi_echelon_exact_summary_is_internally_consistent():
    summary = invman_rust.multi_echelon_exact_dp_summary()
    reference = summary["verification_reference"]

    assert reference["literature_verified"] is False
    assert len(summary["optimal_first_action"]) == 2
    assert len(summary["sequential_first_action"]) == 2
    assert len(summary["proportional_first_action"]) == 2
    assert len(summary["min_shortage_first_action"]) == 2
    assert summary["optimal_discounted_cost"] <= summary["sequential_discounted_cost"]
    assert summary["optimal_discounted_cost"] <= summary["proportional_discounted_cost"]
    assert summary["optimal_discounted_cost"] <= summary["min_shortage_discounted_cost"]


def test_multi_echelon_published_relative_rows_imply_expected_costs():
    references = {
        reference["name"]: reference
        for reference in invman_rust.multi_echelon_list_reference_instances()
        if reference["published_a3c_savings_pct"] is not None
    }

    setting1 = references["van_roy1997_case_study1"]
    setting2 = references["van_roy1997_case_study2"]

    implied_setting1 = setting1["published_constant_base_stock_mean_cost"] * (
        1.0 - setting1["published_a3c_savings_pct"] / 100.0
    )
    implied_setting2 = setting2["published_constant_base_stock_mean_cost"] * (
        1.0 - setting2["published_a3c_savings_pct"] / 100.0
    )

    assert implied_setting1 == pytest.approx(1185.471)
    assert implied_setting2 == pytest.approx(1273.8159)


def test_multi_echelon_gijs_relative_verification_summary_binding():
    summary = invman_rust.multi_echelon_gijs_relative_verification_summary(
        repo_audit_replications=2,
        seed=123,
    )

    assert "Gijsbrechts" in summary["source"]
    assert len(summary["rows"]) == 2
    assert summary["rows"][0]["instance_name"] == "van_roy1997_case_study1"
    assert summary["rows"][0]["published_a3c_implied_mean_cost"] == pytest.approx(
        1185.471
    )
    assert summary["literature_reference_present"] is True
    assert summary["implementation_literature_verified"] is False
    assert (
        summary["literature_verification_metric"]
        == "published_relative_a3c_savings_vs_constant_base_stock_pct"
    )
    assert summary["literature_verification_target_count"] == 2
    assert (
        summary["all_published_constant_base_stock_rows_reproduced_within_tolerance"]
        is True
    )
    assert summary["can_mark_literature_verified"] is False


def test_multi_echelon_van_roy_reproduction_summary_binding():
    summary = invman_rust.multi_echelon_van_roy_reproduction_summary(
        repo_audit_replications=2,
        seed=123,
    )

    assert "Van Roy" in summary["source"]
    assert (
        summary["literature_verification_metric"]
        == "published_constant_base_stock_mean_cost"
    )
    assert summary["literature_reference_present"] is True
    assert summary["implementation_literature_verified"] is False
    assert summary["literature_verification_target_count"] == 3
    assert (
        summary["all_published_constant_base_stock_rows_reproduced_within_tolerance"]
        is True
    )

    rows = {row["instance_name"]: row for row in summary["rows"]}
    assert rows["van_roy1997_simple_problem"][
        "published_constant_base_stock_levels"
    ] == [10, 16]
    assert rows["van_roy1997_case_study1"][
        "published_constant_base_stock_mean_cost"
    ] == pytest.approx(1302.0)
    assert rows["van_roy1997_case_study2"][
        "published_constant_base_stock_mean_cost"
    ] == pytest.approx(1449.0)
    assert (
        abs(rows["van_roy1997_case_study1"]["repo_gap_vs_published_constant_cost_pct"])
        > 1.0
    )


def test_multi_echelon_raw_state_builder_uses_raw_layout():
    raw_state = invman_rust.multi_echelon_build_raw_state(
        warehouse_inventory=3,
        warehouse_pipeline=[2, 2],
        retailer_inventory=[1, 0],
        retailer_pipeline=[[1], [0]],
    )
    assert raw_state == pytest.approx([3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0])
