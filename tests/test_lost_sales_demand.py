import pytest

import invman_rust


def _reference(name):
    return invman_rust.lost_sales_reference_costs(name)


def test_lost_sales_reference_table_carries_iid_demand_families():
    poisson = _reference("lit_poisson_p4_l4")
    geometric = _reference("lit_geometric_p4_l4")

    assert poisson["demand_kind"] == "Poisson"
    assert poisson["demand_rate"] == 5.0
    assert poisson["costs"]["myopic2"] == pytest.approx(4.82)

    assert geometric["demand_kind"] == "Geometric"
    assert geometric["demand_rate"] == 5.0
    assert geometric["costs"]["myopic2"] == pytest.approx(10.8)


def test_lost_sales_reference_table_carries_positive_and_negative_mmpp2_regimes():
    positive = _reference("lit_mmpp2_pos_p4_l4")
    negative = _reference("lit_mmpp2_neg_p4_l4")

    for reference in (positive, negative):
        assert reference["demand_kind"] == "MarkovModulatedPoisson2"
        assert reference["demand_rate"] == 5.0
        assert reference["demand_lambda_low"] == 3.0
        assert reference["demand_lambda_high"] == 7.0
        assert reference["source"] == "computed"
        assert reference["costs"]["optimal"] is None

    assert positive["demand_p00"] == pytest.approx(0.9)
    assert positive["demand_p11"] == pytest.approx(0.9)
    assert negative["demand_p00"] == pytest.approx(0.1)
    assert negative["demand_p11"] == pytest.approx(0.1)
    assert positive["costs"]["myopic2"] > negative["costs"]["myopic2"]


def test_rust_heuristic_evaluator_distinguishes_mmpp2_correlation_regimes():
    positive = invman_rust.lost_sales_heuristics_all(
        "MarkovModulatedPoisson2",
        5.0,
        3.0,
        7.0,
        0.9,
        0.9,
        4,
        1.0,
        4.0,
        0.0,
        0.0,
        200,
        123,
        0.2,
        20,
        1.0,
    )
    negative = invman_rust.lost_sales_heuristics_all(
        "MarkovModulatedPoisson2",
        5.0,
        3.0,
        7.0,
        0.1,
        0.1,
        4,
        1.0,
        4.0,
        0.0,
        0.0,
        200,
        123,
        0.2,
        20,
        1.0,
    )

    assert set(positive) == {"myopic1", "myopic2", "svbs"}
    assert set(negative) == {"myopic1", "myopic2", "svbs"}
    assert positive["myopic2"] > negative["myopic2"]
