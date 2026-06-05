import invman_rust


def test_reference_heuristics_match_literature_values():
    reference = invman_rust.lost_sales_reference_costs("vanilla_l4_p4_poisson5")
    costs = reference["costs"]

    assert costs["myopic2"] < costs["myopic1"] < costs["svbs"]
    assert costs["myopic1"] == 5.06
    assert costs["myopic2"] == 4.82
    assert costs["svbs"] == 5.83
