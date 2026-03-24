from invman.problems.lost_sales_fixed_order_cost.reference_instances import (
    build_grid_instances,
    get_reference_instance,
    list_reference_instances,
)


def test_literature_subset_grid_has_expected_size():
    instances = build_grid_instances("literature_subset_poisson_mu5")
    assert len(instances) == 16


def test_literature_subset_grid_contains_canonical_instance():
    instance = get_reference_instance("lit_pois_mu5_l4_p4_k5")
    assert instance["params"]["lead_time"] == 4
    assert instance["params"]["shortage_cost"] == 4.0
    assert instance["params"]["fixed_order_cost"] == 5.0
    anchors = instance["benchmark_anchors"]
    approximators = anchors["policy_approximator_anchors"]
    assert approximators["linear_categorical_quantity"]["mean_cost"] > approximators["nn_gated_ordinal_quantity"]["mean_cost"]
    assert approximators["nn_gated_ordinal_quantity"]["mean_cost"] > approximators["soft_tree_depth1_linear_leaf"]["mean_cost"]
    assert anchors["heuristic_anchors_1m"]["modified_s_s_q"]["mean_cost"] < 9.2
    assert anchors["learned_policy_anchors"]["current_best_tree_depth1_1m"] < anchors["heuristic_anchors_1m"]["modified_s_s_q"]["mean_cost"]


def test_reference_instance_names_are_sorted_and_stable():
    names = list_reference_instances()
    assert names[0] == "lit_pois_mu5_l1_p19_k25"
    assert names[-1] == "lit_pois_mu5_l4_p4_k5"
