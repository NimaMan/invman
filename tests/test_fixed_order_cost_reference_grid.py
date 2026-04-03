from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    build_reference_args,
    build_grid_instances,
    get_reference_instance,
    list_reference_instances,
)


def test_literature_subset_grid_has_expected_size():
    instances = build_grid_instances("literature_subset_poisson_mu5")
    assert len(instances) == 16


def test_correlated_mmpp2_grid_has_positive_and_negative_cases():
    instances = build_grid_instances("correlated_mmpp2_mu5_l4_p4_k5")
    assert len(instances) == 2
    names = {instance["name"] for instance in instances}
    assert names == {"corr_mmpp2_neg_mu5_l4_p4_k5", "corr_mmpp2_pos_mu5_l4_p4_k5"}
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
    anchors = instance["benchmark_anchors"]
    approximators = anchors["policy_approximator_anchors"]
    assert approximators["linear_categorical_quantity"]["mean_cost"] > approximators["linear_soft_gated_ordinal_quantity"]["mean_cost"]
    assert approximators["nn_categorical_quantity"]["verification_status"] == "needs_verification"
    assert approximators["nn_soft_gated_ordinal_quantity"]["mean_cost"] < approximators["linear_soft_gated_ordinal_quantity"]["mean_cost"]
    assert approximators["linear_soft_gated_ordinal_quantity"]["mean_cost"] < approximators["soft_tree_depth1_linear_leaf"]["mean_cost"]
    assert anchors["heuristic_anchors_1m"]["modified_s_s_q"]["mean_cost"] < 9.2
    assert approximators["nn_soft_gated_ordinal_quantity"]["mean_cost"] < anchors["heuristic_anchors_1m"]["modified_s_s_q"]["mean_cost"]


def test_reference_instance_names_are_sorted_and_stable():
    names = list_reference_instances()
    assert "bijvank2015_table1_l2_p14_k5" in names
    assert names[0] == "bijvank2015_table1_l2_p14_k5"
    assert names[-1] == "lit_pois_mu5_l4_p4_k5"


def test_published_validation_instance_matches_reported_benchmark():
    instance = get_reference_instance("bijvank2015_table1_l2_p14_k5")
    assert instance["params"]["lead_time"] == 2
    assert instance["params"]["shortage_cost"] == 14.0
    assert instance["params"]["fixed_order_cost"] == 5.0
    published = instance["benchmark_anchors"]["published_heuristic_references"]
    assert published["s_s"]["params"] == {"s": 17, "S": 23}
    assert published["s_nq"]["params"] == {"s": 17, "q": 7}
    assert published["modified_s_s_q"]["params"] == {"s": 17, "S": 23, "q": 7}
    assert instance["benchmark_anchors"]["published_optimal_reference"]["mean_cost"] == 11.46


def test_fixed_cost_reference_args_do_not_force_tracked_demands():
    args = build_reference_args("lit_pois_mu5_l4_p4_k5")
    assert not getattr(args, "track_demand", False)
