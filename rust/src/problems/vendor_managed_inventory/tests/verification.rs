use crate::problems::vendor_managed_inventory::env::{
    build_policy_state, initialize_state, step_state, terminal_salvage_credit,
};
use crate::problems::vendor_managed_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::vendor_managed_inventory::heuristics::{
    dc_reserve_base_stock_shipment_quantity, retailer_base_stock_shipment_quantity,
};
use crate::problems::vendor_managed_inventory::references::{
    GIANNOCCARO_2010_REFERENCE, PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
    WORKED_TRANSITION_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(
        GIANNOCCARO_2010_REFERENCE.benchmark_policies,
        &["retailer_base_stock", "projected_order_up_to", "rl_policy"]
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.dc_capacity, 10);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.benchmark_dc_reserve_quantity, 2);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_shipment_quantity, 4);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_dc_on_hand,
        VERIFICATION_PROBLEM_INSTANCE.initial_retailer_on_hand,
        VERIFICATION_PROBLEM_INSTANCE.initial_retailer_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.dc_capacity,
    )
    .expect("state must build");
    let features = build_policy_state(
        &state,
        2.5,
        VERIFICATION_PROBLEM_INSTANCE.periods,
        VERIFICATION_PROBLEM_INSTANCE.dc_capacity,
        VERIFICATION_PROBLEM_INSTANCE.dc_replenishment_quantity,
    )
    .expect("policy state must build");

    assert_eq!(features.len(), 7);
    assert!((features[0] - 0.8).abs() < 1e-6);
    assert!((features[1] - 0.2).abs() < 1e-6);
    assert!((features[2] - 0.2).abs() < 1e-6);
    assert!((features[3] - 0.4).abs() < 1e-6);
    assert!((features[4] - 0.5).abs() < 1e-6);
    assert!((features[5] - 0.4).abs() < 1e-6);
    assert!((features[6] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_dc_on_hand,
        worked.initial_retailer_on_hand,
        worked.initial_retailer_pipeline,
        worked.dc_capacity,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.shipment_quantity,
        worked.realized_demand,
        worked.dc_replenishment_quantity,
        worked.dc_capacity,
        worked.shipment_cost_per_unit,
        worked.dc_holding_cost_per_unit,
        worked.retailer_holding_cost_per_unit,
        worked.stockout_cost_per_unit,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.arrivals_to_retailer,
        worked.expected_arrivals_to_retailer
    );
    assert_eq!(outcome.sales, worked.expected_sales);
    assert_eq!(outcome.lost_sales, worked.expected_lost_sales);
    assert_eq!(outcome.dc_replenishment, worked.expected_dc_replenishment);
    assert_eq!(
        outcome.next_state.dc_on_hand,
        worked.expected_next_dc_on_hand
    );
    assert_eq!(
        outcome.next_state.retailer_on_hand,
        worked.expected_next_retailer_on_hand
    );
    assert_eq!(
        outcome.next_state.retailer_pipeline,
        worked.expected_next_retailer_pipeline
    );
    assert!((outcome.period_cost - worked.expected_period_cost).abs() < 1e-12);
}

#[test]
fn terminal_salvage_credit_matches_expected_freeze() {
    let state = initialize_state(2, 1, 3, 5).expect("state must build");
    let credit = terminal_salvage_credit(&state, 5, 0.2).expect("terminal credit must compute");
    assert!((credit - 1.2).abs() < 1e-12);
}

#[test]
fn heuristic_first_actions_match_reference_freeze() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_dc_on_hand,
        VERIFICATION_PROBLEM_INSTANCE.initial_retailer_on_hand,
        VERIFICATION_PROBLEM_INSTANCE.initial_retailer_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.dc_capacity,
    )
    .expect("state must build");
    let retailer_base_stock = retailer_base_stock_shipment_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.retailer_base_stock_level,
        VERIFICATION_PROBLEM_INSTANCE.max_shipment_quantity,
    )
    .expect("retailer base-stock must compute");
    let dc_reserve = dc_reserve_base_stock_shipment_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.dc_reserve_base_stock_level,
        VERIFICATION_PROBLEM_INSTANCE.dc_reserve_quantity,
        VERIFICATION_PROBLEM_INSTANCE.max_shipment_quantity,
    )
    .expect("dc-reserve base-stock must compute");

    assert_eq!(
        retailer_base_stock,
        VERIFICATION_PROBLEM_INSTANCE.expected_retailer_base_stock_first_action
    );
    assert_eq!(
        dc_reserve,
        VERIFICATION_PROBLEM_INSTANCE.expected_dc_reserve_base_stock_first_action
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let retailer_base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "retailer_base_stock")
            .expect("retailer base-stock evaluation must solve");
    let dc_reserve =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "dc_reserve_base_stock")
            .expect("dc-reserve base-stock evaluation must solve");

    assert!(
        (optimal.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
            .abs()
            < 1e-9,
        "optimal discounted cost freeze mismatch: got {}",
        optimal.discounted_cost
    );
    assert_eq!(
        optimal.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action
    );
    assert!(
        (retailer_base_stock.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_retailer_base_stock_discounted_cost)
            .abs()
            < 1e-9,
        "retailer_base_stock discounted cost freeze mismatch: got {}",
        retailer_base_stock.discounted_cost
    );
    assert_eq!(
        retailer_base_stock.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_retailer_base_stock_first_action
    );
    assert!(
        (dc_reserve.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_dc_reserve_base_stock_discounted_cost)
            .abs()
            < 1e-9,
        "dc_reserve_base_stock discounted cost freeze mismatch: got {}",
        dc_reserve.discounted_cost
    );
    assert_eq!(
        dc_reserve.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_dc_reserve_base_stock_first_action
    );
}
