from invman.reference_instances import VANILLA_L4_P4_POISSON5, evaluate_reference_heuristics


def test_reference_heuristics_match_literature_values():
    results = evaluate_reference_heuristics(
        name=VANILLA_L4_P4_POISSON5.name,
        horizon=int(1e5),
        seeds=[123],
    )

    assert results["myopic2"]["mean_cost"] < results["myopic1"]["mean_cost"] < results["svbs"]["mean_cost"]
    assert abs(results["myopic1"]["mean_cost"] - VANILLA_L4_P4_POISSON5.expected_costs["myopic1"]) <= 0.08
    assert abs(results["myopic2"]["mean_cost"] - VANILLA_L4_P4_POISSON5.expected_costs["myopic2"]) <= 0.03
    assert abs(results["svbs"]["mean_cost"] - VANILLA_L4_P4_POISSON5.expected_costs["svbs"]) <= 0.03
