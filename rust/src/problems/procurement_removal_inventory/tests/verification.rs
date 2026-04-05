use crate::problems::procurement_removal_inventory::env::{
    build_policy_state, initialize_state, step_state, terminal_salvage_credit,
};
use crate::problems::procurement_removal_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, returnability_buffer_interval_stock_action,
};
use crate::problems::procurement_removal_inventory::references::{
    MAGGIAR_2017_REFERENCE, MAGGIAR_2025_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(
        MAGGIAR_2017_REFERENCE.benchmark_policies,
        &[
            "optimal_interval_stock",
            "order_up_to_remove_down_to",
            "pricing_and_markdown_variants"
        ]
    );
    assert_eq!(
        MAGGIAR_2025_REFERENCE.benchmark_policies,
        &["directbackprop_drl", "structure_informed_policy_network", "interval_stock"]
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.returnable_purchase_cap, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.benchmark_returnable_buffer, 2);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity, 4);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity, 4);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.initial_returnable_inventory,
    )
    .expect("state must build");
    let features = build_policy_state(
        &state,
        1.7,
        VERIFICATION_PROBLEM_INSTANCE.periods,
        VERIFICATION_PROBLEM_INSTANCE.returnable_purchase_cap,
    )
    .expect("policy state must build");

    assert_eq!(features.len(), 7);
    assert!((features[0] - 1.0).abs() < 1e-6);
    assert!((features[1] - 0.5).abs() < 1e-6);
    assert!((features[2] - 0.5).abs() < 1e-6);
    assert!((features[3] - 0.5).abs() < 1e-6);
    assert!((features[4] - 0.85).abs() < 1e-6);
    assert!((features[5] - 0.5).abs() < 1e-6);
    assert!((features[6] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_inventory_level,
        worked.initial_returnable_inventory,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.purchase_quantity,
        worked.removal_quantity,
        worked.realized_demand,
        worked.returnable_purchase_cap,
        worked.purchase_cost_per_unit,
        worked.return_value_per_unit,
        worked.liquidation_value_per_unit,
        worked.holding_cost_per_unit,
        worked.shortage_cost_per_unit,
    )
    .expect("step must succeed");

    assert_eq!(outcome.returned_units, worked.expected_returned_units);
    assert_eq!(outcome.liquidated_units, worked.expected_liquidated_units);
    assert_eq!(outcome.sales, worked.expected_sales);
    assert_eq!(outcome.shortage, worked.expected_shortage);
    assert_eq!(
        outcome.next_state.inventory_level,
        worked.expected_next_inventory_level
    );
    assert_eq!(
        outcome.next_state.returnable_inventory,
        worked.expected_next_returnable_inventory
    );
    assert!((outcome.period_cost - worked.expected_period_cost).abs() < 1e-12);
}

#[test]
fn terminal_salvage_credit_matches_expected_freeze() {
    let state = initialize_state(3, 1).expect("state must build");
    let credit = terminal_salvage_credit(&state, 3.0, 1.0).expect("terminal credit must compute");
    assert!((credit - 5.0).abs() < 1e-12);
}

#[test]
fn heuristic_first_actions_match_reference_freeze() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.initial_returnable_inventory,
    )
    .expect("state must build");
    let interval = interval_stock_action(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.interval_stock_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.interval_stock_remove_down_to,
        VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity,
        VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity,
    )
    .expect("interval-stock must compute");
    let buffer = returnability_buffer_interval_stock_action(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer_remove_down_to,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer,
        VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity,
        VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity,
    )
    .expect("buffered interval-stock must compute");

    assert_eq!(
        interval,
        VERIFICATION_PROBLEM_INSTANCE.expected_interval_stock_first_action
    );
    assert_eq!(
        buffer,
        VERIFICATION_PROBLEM_INSTANCE.expected_returnability_buffer_first_action
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let interval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "interval_stock")
        .expect("interval-stock evaluation must solve");
    let buffer = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "returnability_buffer_interval_stock",
    )
    .expect("buffered interval-stock evaluation must solve");

    assert!(
        (optimal.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        optimal.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action
    );
    assert!(
        (interval.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_interval_stock_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        interval.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_interval_stock_first_action
    );
    assert!(
        (buffer.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_returnability_buffer_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        buffer.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_returnability_buffer_first_action
    );
}
