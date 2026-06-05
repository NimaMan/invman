import pytest

import invman_rust

from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import (
    benchmark_reference_instance,
    build_reference_args,
    get_benchmark_grid,
    get_reference_instance,
)


def test_full_grid_has_expected_size_and_axes():
    instances = get_benchmark_grid("lost_sales_style_full_grid_mu5")["instances"]
    assert len(instances) == 64
    demand_names = {instance["params"]["demand_dist_name"] for instance in instances}
    assert demand_names == {"Poisson", "Geometric", "MarkovModulatedPoisson2"}
    lead_times = {instance["params"]["lead_time"] for instance in instances}
    assert lead_times == {4, 6, 8, 10}
    fixed_costs = {instance["params"]["fixed_order_cost"] for instance in instances}
    assert fixed_costs == {5.0, 25.0}


def test_correlated_mmpp2_grid_has_positive_and_negative_cases():
    instances = [
        instance
        for instance in get_benchmark_grid("lost_sales_style_full_grid_mu5")["instances"]
        if instance["params"]["lead_time"] == 4
        and instance["params"]["shortage_cost"] == 4.0
        and instance["params"]["fixed_order_cost"] == 5.0
        and instance["params"]["demand_dist_name"] == "MarkovModulatedPoisson2"
    ]
    assert len(instances) == 2
    names = {instance["name"] for instance in instances}
    assert names == {"lit_mmpp2_neg_mu5_l4_p4_k5", "lit_mmpp2_pos_mu5_l4_p4_k5"}
    for instance in instances:
        params = instance["params"]
        assert params["demand_dist_name"] == "MarkovModulatedPoisson2"
        assert params["demand_rate"] == 5.0
        assert params["demand_lambda_low"] == 3.0
        assert params["demand_lambda_high"] == 7.0


def test_literature_subset_grid_contains_canonical_instance():
    instance = get_reference_instance("lit_pois_mu5_l4_p4_k5")
    assert instance["params"]["lead_time"] == 4
    assert instance["params"]["shortage_cost"] == 4.0
    assert instance["params"]["fixed_order_cost"] == 5.0
    assert instance["search"]["position_upper_bound"] == 45
    assert instance["literature_metadata"]["parent_problem_family"] == "Bijvank2015ParametricPolicies"


def test_reference_instance_names_are_sorted_and_stable():
    names = invman_rust.lost_sales_fixed_order_cost_list_reference_instances()
    assert "bijvank2015_table1_l2_p14_k5" in names
    assert names[0] == "bijvank2015_table1_l2_p14_k5"


def test_published_validation_instance_matches_reported_benchmark():
    summary = invman_rust.lost_sales_fixed_order_cost_exact_literature_summary()
    instance = summary["reference"]
    assert instance["lead_time"] == 2
    assert instance["shortage_cost"] == 14.0
    assert instance["fixed_order_cost"] == 5.0
    published = {
        row["policy_name"]: row for row in instance["published_heuristic_rows"]
    }
    assert published["s_s"]["params"] == [17, 23]
    assert published["s_nq"]["params"] == [17, 7]
    assert published["modified_s_s_q"]["params"] == [17, 23, 7]
    assert summary["published_optimal_cost"] == 11.46
    assert summary["optimal_average_cost"] == pytest.approx(11.463052002030395)


def test_fixed_cost_reference_args_do_not_force_tracked_demands():
    args = build_reference_args("lit_pois_mu5_l4_p4_k5")
    assert not getattr(args, "track_demand", False)


def test_benchmark_reference_instance_reports_available_rust_heuristic_baselines():
    payload = benchmark_reference_instance(
        "lit_pois_mu5_l4_p4_k5",
        search_horizon=100,
        search_seed=123,
    )

    assert payload["note"].startswith("fixed-cost heuristic baselines evaluated")
    assert payload["capped_base_stock_reference"]["source"] == "not_applicable_fixed_order_cost"
    assert set(payload["evaluation"]) == {"s_s", "s_nq", "modified_s_s_q"}
    expected_param_lengths = {"s_s": 2, "s_nq": 2, "modified_s_s_q": 3}
    for policy_name, row in payload["evaluation"].items():
        assert row["available"] is True
        assert row["source"] == "rust_lost_sales_fixed_heuristics_all_detailed"
        assert row["mean_cost"] > 0.0
        assert len(row["params"]) == expected_param_lengths[policy_name]
        assert row["top"][0]["params"] == row["params"]
        assert row["top"][0]["mean_cost"] == pytest.approx(row["mean_cost"])
        assert row["search_horizon"] == 100
        assert row["search_seed"] == 123


def test_fixed_heuristics_detailed_binding_matches_cost_only_binding():
    kwargs = dict(
        demand_kind="Poisson",
        demand_rate=5.0,
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
        lead_time=4,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        max_order_size=20,
        position_upper_bound=30,
        horizon=50,
        seed=123,
        warm_up_periods_ratio=0.2,
        top_k=2,
    )

    cost_only = dict(invman_rust.lost_sales_fixed_heuristics_all(**kwargs))
    detailed = dict(invman_rust.lost_sales_fixed_heuristics_all_detailed(**kwargs))

    for policy_name, expected_params_len in {
        "s_s": 2,
        "s_nq": 2,
        "modified_s_s_q": 3,
    }.items():
        row = detailed[policy_name]
        assert row["mean_cost"] == pytest.approx(cost_only[policy_name])
        assert len(row["params"]) == expected_params_len
        assert len(row["top"]) == 2
        assert row["top"][0]["params"] == row["params"]
