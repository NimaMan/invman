use crate::problems::procurement_removal_inventory::env::{
    build_raw_state, initialize_state, step_state, terminal_salvage_credit,
};
use crate::problems::procurement_removal_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, returnability_buffer_interval_stock_action,
};
use crate::problems::procurement_removal_inventory::literature::references::{
    MAGGIAR_2017_REFERENCE, MAGGIAR_2025_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    REMOVAL_ACTIVE_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
};

#[derive(Clone, Copy)]
struct WorkedTransitionCase {
    initial_inventory_level: usize,
    initial_returnable_inventory: usize,
    purchase_quantity: usize,
    removal_quantity: usize,
    realized_demand: usize,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    expected_returned_units: usize,
    expected_liquidated_units: usize,
    expected_sales: usize,
    expected_shortage: usize,
    expected_next_inventory_level: usize,
    expected_next_returnable_inventory: usize,
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_inventory_level: 4,
    initial_returnable_inventory: 2,
    purchase_quantity: 3,
    removal_quantity: 2,
    realized_demand: 4,
    returnable_purchase_cap: 2,
    purchase_cost_per_unit: 6.0,
    return_value_per_unit: 4.0,
    liquidation_value_per_unit: 1.0,
    holding_cost_per_unit: 0.5,
    shortage_cost_per_unit: 9.0,
    expected_returned_units: 2,
    expected_liquidated_units: 0,
    expected_sales: 4,
    expected_shortage: 0,
    expected_next_inventory_level: 1,
    expected_next_returnable_inventory: 1,
    expected_period_cost: 10.5,
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
        &[
            "directbackprop_drl",
            "structure_informed_policy_network",
            "interval_stock"
        ]
    );
    assert!(!MAGGIAR_2017_REFERENCE.reported_numbers_available);
    assert!(!MAGGIAR_2017_REFERENCE.numbers_anchor_repo_assertions);
    assert!(!MAGGIAR_2025_REFERENCE.reported_numbers_available);
    assert!(!MAGGIAR_2025_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.returnable_purchase_cap, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.benchmark_returnable_buffer, 2);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity, 4);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity, 4);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.initial_returnable_inventory,
    )
    .expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");

    assert_eq!(raw_state, vec![2.0, 1.0, 0.0]);
}

#[test]
fn raw_state_preserves_high_inventory_magnitude() {
    let state = initialize_state(8, 3).expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");

    assert_eq!(raw_state, vec![8.0, 3.0, 0.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
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
fn heuristic_first_actions_match_named_heuristic_evaluators() {
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

    let interval_eval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "interval_stock")
        .expect("interval-stock evaluation must solve");
    let buffer_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "returnability_buffer_interval_stock",
    )
    .expect("buffered interval-stock evaluation must solve");

    assert_eq!(interval, interval_eval.first_action);
    assert_eq!(buffer, buffer_eval.first_action);
}

#[test]
fn removal_active_instance_is_well_formed() {
    // The removal-active benchmark instance must (a) be a valid env state and (b) respect the
    // Maggiar & Sadighian (2017) cost ordering used throughout this package: Assumption 2(ii)
    // c > s (purchase cost above return value) and 2(iii) l <= s (liquidation no greater than
    // return value). It must also start overstocked relative to demand so the removal channel can
    // bind, and its carried benchmark levels must form a valid interval (order_up_to <=
    // remove_down_to) where the removal level is strictly above the order level (removal lever is
    // exercised), unlike the primary instance where they collapse together.
    let instance = REMOVAL_ACTIVE_REFERENCE_INSTANCE;
    let state = initialize_state(
        instance.initial_inventory_level,
        instance.initial_returnable_inventory,
    )
    .expect("removal-active initial state must build");
    assert!(state.returnable_inventory <= state.inventory_level);

    assert!(instance.purchase_cost_per_unit > instance.return_value_per_unit);
    assert!(instance.return_value_per_unit >= instance.liquidation_value_per_unit);

    assert!(instance.benchmark_order_up_to <= instance.benchmark_remove_down_to);
    assert!(instance.benchmark_remove_down_to > instance.benchmark_order_up_to);
    assert!((instance.initial_inventory_level as f64) > instance.demand_mean);

    // A worked step that exercises the removal channel: from the overstocked start, removing units
    // returns from the returnable pool first (Corollary 1: never liquidate what can be returned),
    // and any excess beyond the returnable pool is liquidated.
    let outcome = step_state(
        &state,
        0,  // no purchase
        10, // remove 10 of 12 on hand: 8 returnable + 2 liquidated
        0,  // zero demand to isolate the removal accounting
        instance.returnable_purchase_cap,
        instance.purchase_cost_per_unit,
        instance.return_value_per_unit,
        instance.liquidation_value_per_unit,
        instance.holding_cost_per_unit,
        instance.shortage_cost_per_unit,
    )
    .expect("removal-active worked step must succeed");
    assert_eq!(outcome.returned_units, 8);
    assert_eq!(outcome.liquidated_units, 2);
    assert_eq!(outcome.next_state.inventory_level, 2);
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
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
        optimal.discounted_cost <= interval.discounted_cost + 1e-9,
        "optimal={} interval_stock={}",
        optimal.discounted_cost,
        interval.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= buffer.discounted_cost + 1e-9,
        "optimal={} returnability_buffer={}",
        optimal.discounted_cost,
        buffer.discounted_cost
    );
}
