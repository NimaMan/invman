use crate::problems::joint_replenishment::env::{build_policy_state, initialize_state, step_state};
use crate::problems::joint_replenishment::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::joint_replenishment::heuristics::{
    dynamic_order_up_to_order_quantities, minimum_order_quantity_order_quantities,
};
use crate::problems::joint_replenishment::references::{
    PRIMARY_REFERENCE_INSTANCE, SMALL_SCALE_SETTINGS, VANVUCHELEN_2020_REFERENCE,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(VANVUCHELEN_2020_REFERENCE.benchmark_policies.len(), 3);
    assert!(VANVUCHELEN_2020_REFERENCE.reported_numbers_available);
    assert!(!VANVUCHELEN_2020_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(SMALL_SCALE_SETTINGS.len(), 16);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_items, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.truck_capacity, 6);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.minor_order_costs[0], 40.0);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.demand_ranges[0].high, 5);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.demand_ranges[1].high, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.periods, 4);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(&[2, -1]).expect("state must build");
    let features = build_policy_state(&state, 4).expect("policy state must build");
    assert_eq!(features.len(), 4);
    assert!((features[0] - 1.0).abs() < 1e-6);
    assert!((features[1] + 0.5).abs() < 1e-6);
    assert!((features[2] - 0.5).abs() < 1e-6);
    assert!((features[3] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(worked.initial_inventory_levels).expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_demands,
        worked.truck_capacity,
        worked.minor_order_costs,
        worked.major_order_cost,
        worked.holding_costs,
        worked.shortage_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.next_state.inventory_levels,
        worked.expected_next_inventory_levels.to_vec()
    );
    assert_eq!(outcome.trucks_used, worked.expected_trucks_used);
    assert_eq!(outcome.order_cost, worked.expected_order_cost);
    assert_eq!(outcome.holding_cost, worked.expected_holding_cost);
    assert_eq!(outcome.shortage_cost, worked.expected_shortage_cost);
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
    assert_eq!(outcome.reward, -worked.expected_period_cost);
}

#[test]
fn heuristic_initial_orders_match_reference_freeze() {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let state = initialize_state(reference.initial_inventory_levels).expect("state must build");
    let moq_action = minimum_order_quantity_order_quantities(
        &state,
        reference.moq_item_targets,
        reference.moq_review_period,
        reference.moq_rounding_threshold,
        reference.truck_capacity,
    )
    .expect("MOQ heuristic must succeed");
    let dynout_action = dynamic_order_up_to_order_quantities(
        &state,
        reference.dynout_item_targets,
        reference.truck_capacity,
        reference.demand_ranges,
        reference.holding_costs,
        reference.shortage_costs,
    )
    .expect("DYN-OUT heuristic must succeed");

    assert_eq!(moq_action, reference.expected_moq_first_action.to_vec());
    assert_eq!(
        dynout_action,
        reference.expected_dynout_first_action.to_vec()
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let moq = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "minimum_order_quantity")
        .expect("MOQ evaluation must solve");
    let dynout = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "dynamic_order_up_to")
        .expect("DYN-OUT evaluation must solve");

    assert!(
        (optimal.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        optimal.first_action,
        [
            VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action[0],
            VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action[1]
        ]
    );

    assert!(
        (moq.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_moq_discounted_cost).abs()
            < 1e-9
    );
    assert_eq!(
        moq.first_action,
        [
            VERIFICATION_PROBLEM_INSTANCE.expected_moq_first_action[0],
            VERIFICATION_PROBLEM_INSTANCE.expected_moq_first_action[1]
        ]
    );

    assert!(
        (dynout.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_dynout_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        dynout.first_action,
        [
            VERIFICATION_PROBLEM_INSTANCE.expected_dynout_first_action[0],
            VERIFICATION_PROBLEM_INSTANCE.expected_dynout_first_action[1]
        ]
    );
}
