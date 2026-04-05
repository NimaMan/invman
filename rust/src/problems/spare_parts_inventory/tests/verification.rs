use crate::problems::spare_parts_inventory::env::{
    build_policy_state, initialize_state, step_state,
};
use crate::problems::spare_parts_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::spare_parts_inventory::heuristics::{
    base_stock_order_quantity, lead_time_mean_cover_order_quantity,
};
use crate::problems::spare_parts_inventory::references::{
    PRIMARY_REFERENCE_INSTANCE, SPARE_PARTS_REVIEW_REFERENCE, VAN_DER_HAAR_2025_REFERENCE,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE, ZHOU_2024_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(SPARE_PARTS_REVIEW_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(ZHOU_2024_REFERENCE.benchmark_policies, &["marl", "multi-echelon spare-parts baselines"]);
    assert_eq!(
        VAN_DER_HAAR_2025_REFERENCE.benchmark_policies,
        &["drl", "distance-based transshipment and expediting heuristics"]
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.installed_base, 12);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.procurement_lead_time, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.installed_base, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantity, 4);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        VERIFICATION_PROBLEM_INSTANCE.initial_procurement_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.initial_repair_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
    )
    .expect("state must build");
    let features = build_policy_state(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
        VERIFICATION_PROBLEM_INSTANCE.failure_probability,
        VERIFICATION_PROBLEM_INSTANCE.periods,
    )
    .expect("policy state must build");

    assert_eq!(features.len(), 11);
    assert!((features[0] - 0.33333334).abs() < 1e-6);
    assert!((features[2] - 0.33333334).abs() < 1e-6);
    assert!((features[3] - 1.0).abs() < 1e-6);
    assert!((features[9] - 0.4).abs() < 1e-6);
    assert!((features[10] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        worked.initial_procurement_pipeline,
        worked.initial_repair_pipeline,
        worked.installed_base,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_failures,
        worked.installed_base,
        worked.holding_cost,
        worked.downtime_cost,
        worked.procurement_cost,
    )
    .expect("step must succeed");

    assert_eq!(outcome.procurement_arrival, worked.expected_procurement_arrival);
    assert_eq!(outcome.repair_return, worked.expected_repair_return);
    assert_eq!(
        outcome.post_failure_on_hand_inventory,
        worked.expected_post_failure_on_hand_inventory
    );
    assert_eq!(outcome.post_failure_backlog, worked.expected_post_failure_backlog);
    assert_eq!(
        outcome.next_state.on_hand_inventory,
        worked.expected_next_on_hand_inventory
    );
    assert_eq!(outcome.next_state.backlog, worked.expected_next_backlog);
    assert_eq!(
        outcome.next_state.procurement_pipeline,
        worked.expected_next_procurement_pipeline.to_vec()
    );
    assert_eq!(
        outcome.next_state.repair_pipeline,
        worked.expected_next_repair_pipeline.to_vec()
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn heuristic_first_actions_match_reference_freeze() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        VERIFICATION_PROBLEM_INSTANCE.initial_procurement_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.initial_repair_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
    )
    .expect("state must build");
    let base_stock =
        base_stock_order_quantity(&state, VERIFICATION_PROBLEM_INSTANCE.base_stock_level)
            .expect("base-stock must compute");
    let mean_cover = lead_time_mean_cover_order_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
        VERIFICATION_PROBLEM_INSTANCE.failure_probability,
        VERIFICATION_PROBLEM_INSTANCE.lead_time_mean_cover_safety_buffer,
    )
    .expect("lead-time mean-cover must compute");

    assert_eq!(
        base_stock,
        VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_first_action
    );
    assert_eq!(
        mean_cover,
        VERIFICATION_PROBLEM_INSTANCE.expected_lead_time_mean_cover_first_action
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let mean_cover =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "lead_time_mean_cover")
            .expect("lead-time mean-cover evaluation must solve");

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
        (base_stock.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        base_stock.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_first_action
    );
    assert!(
        (mean_cover.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_lead_time_mean_cover_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        mean_cover.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_lead_time_mean_cover_first_action
    );
}
