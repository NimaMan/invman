use crate::problems::joint_pricing_inventory::env::{
    build_raw_state, initialize_state, step_state, terminal_salvage_credit,
    JointPricingInventoryState,
};
use crate::problems::joint_pricing_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::joint_pricing_inventory::heuristics::{
    inventory_sensitive_base_stock_action, static_price_base_stock_action,
};
use crate::problems::joint_pricing_inventory::references::{
    PRIMARY_REFERENCE_INSTANCE, QIN_2022_REFERENCE, VERIFICATION_PROBLEM_INSTANCE,
    WORKED_TRANSITION_REFERENCE, ZHOU_2022_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(
        ZHOU_2022_REFERENCE.benchmark_policies,
        &[
            "ddqn_joint_price_inventory",
            "value_iteration_baseline",
            "q_learning_baseline"
        ]
    );
    assert_eq!(
        QIN_2022_REFERENCE.benchmark_policies,
        &[
            "data_driven_approximation",
            "deterministic_baseline",
            "random_baseline"
        ]
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.price_levels.len(), 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantity, 4);
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state = JointPricingInventoryState {
        period: 2,
        inventory_level: 3,
    };
    let raw_state = build_raw_state(&state);
    assert_eq!(raw_state, vec![3.0, 2.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(worked.initial_inventory_level).expect("state must build");
    let outcome = step_state(
        &state,
        worked.order_quantity,
        worked.price_index,
        worked.realized_demand,
        worked.price_levels,
        worked.procurement_cost_per_unit,
        worked.holding_cost_per_unit,
        worked.stockout_cost_per_unit,
    )
    .expect("step must succeed");

    assert_eq!(outcome.sales, worked.expected_sales);
    assert_eq!(outcome.lost_sales, worked.expected_lost_sales);
    assert_eq!(
        outcome.next_state.inventory_level,
        worked.expected_next_inventory_level
    );
    assert!((outcome.period_cost - worked.expected_period_cost).abs() < 1e-12);
}

#[test]
fn terminal_salvage_credit_matches_expected_freeze() {
    let state = initialize_state(3).expect("state must build");
    let credit = terminal_salvage_credit(&state, 1.0).expect("terminal credit must compute");
    assert!((credit - 3.0).abs() < 1e-12);
}

#[test]
fn heuristic_first_actions_match_named_heuristic_evaluators() {
    let state = initialize_state(VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level)
        .expect("state must build");
    let static_policy = static_price_base_stock_action(
        state.inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.static_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.static_price_index,
        VERIFICATION_PROBLEM_INSTANCE.max_order_quantity,
        VERIFICATION_PROBLEM_INSTANCE.price_levels.len(),
    )
    .expect("static heuristic must compute");
    let inventory_sensitive = inventory_sensitive_base_stock_action(
        state.inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.inventory_sensitive_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.markdown_threshold,
        VERIFICATION_PROBLEM_INSTANCE.high_price_index,
        VERIFICATION_PROBLEM_INSTANCE.low_price_index,
        VERIFICATION_PROBLEM_INSTANCE.max_order_quantity,
        VERIFICATION_PROBLEM_INSTANCE.price_levels.len(),
    )
    .expect("inventory-sensitive heuristic must compute");

    let static_eval =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "static_price_base_stock")
            .expect("static heuristic evaluation must solve");
    let inventory_sensitive_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "inventory_sensitive_base_stock",
    )
    .expect("inventory-sensitive heuristic evaluation must solve");

    assert_eq!(static_policy, static_eval.first_action);
    assert_eq!(inventory_sensitive, inventory_sensitive_eval.first_action);
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let static_policy =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "static_price_base_stock")
            .expect("static heuristic evaluation must solve");
    let inventory_sensitive = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "inventory_sensitive_base_stock",
    )
    .expect("inventory-sensitive heuristic evaluation must solve");

    assert!(optimal.first_action.0 <= VERIFICATION_PROBLEM_INSTANCE.max_order_quantity);
    assert!(optimal.first_action.1 < VERIFICATION_PROBLEM_INSTANCE.price_levels.len());
    assert!(
        optimal.discounted_cost <= static_policy.discounted_cost + 1e-9,
        "optimal={} static={}",
        optimal.discounted_cost,
        static_policy.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= inventory_sensitive.discounted_cost + 1e-9,
        "optimal={} inventory_sensitive={}",
        optimal.discounted_cost,
        inventory_sensitive.discounted_cost
    );
}
