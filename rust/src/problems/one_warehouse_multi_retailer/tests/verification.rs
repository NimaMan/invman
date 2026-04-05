use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    build_policy_state, initialize_state, retailer_inventory_positions, step_state,
};
use crate::problems::one_warehouse_multi_retailer::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::echelon_base_stock_orders;
use crate::problems::one_warehouse_multi_retailer::references::{
    KAYNOV_2024_REFERENCE, PRIMARY_REFERENCE_INSTANCE, TABLE_A3_INSTANCES,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

fn nested_pipeline_vec(pipelines: &[&[usize]]) -> Vec<Vec<usize>> {
    pipelines.iter().map(|pipeline| pipeline.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(KAYNOV_2024_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(TABLE_A3_INSTANCES.len(), 14);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.name, "kaynov2024_instance_7");
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.retailer_lead_times.len(), 3);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE
            .published_proportional_benchmark
            .expect("primary benchmark must exist")
            .mean_cost,
        -1406.27
    );
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.retailer_lead_times, &[1, 1]);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(3, &[2, 2], &[1, 0], &vec![vec![1], vec![0]])
        .expect("state must build");
    let features = build_policy_state(&state, 4).expect("policy state must build");
    assert_eq!(features.len(), 9);
    assert!((features[0] - 0.33333334).abs() < 1e-6);
    assert!((features[7] - 1.0).abs() < 1e-6);
    assert!((features[8] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_warehouse_inventory,
        worked.initial_warehouse_pipeline,
        worked.initial_retailer_inventory,
        &nested_pipeline_vec(worked.initial_retailer_pipeline),
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.warehouse_order,
        worked.retailer_shipments,
        worked.realized_demands,
        0.5,
        &[1.0, 1.0],
        &[9.0, 9.0],
        worked.customer_behavior,
        0.0,
        None,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.next_state.warehouse_inventory,
        worked.expected_next_warehouse_inventory
    );
    assert_eq!(
        outcome.next_state.warehouse_pipeline,
        worked.expected_next_warehouse_pipeline.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_inventory,
        worked.expected_next_retailer_inventory.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_pipeline,
        nested_pipeline_vec(worked.expected_next_retailer_pipeline)
    );
    assert_eq!(outcome.holding_cost, worked.expected_holding_cost);
    assert_eq!(outcome.shortage_cost, worked.expected_shortage_cost);
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn allocation_and_base_stock_orders_match_reference_freeze() {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let state = initialize_state(
        reference.initial_warehouse_inventory,
        reference.initial_warehouse_pipeline,
        reference.initial_retailer_inventory,
        &nested_pipeline_vec(reference.initial_retailer_pipeline),
    )
    .expect("state must build");
    let action = echelon_base_stock_orders(
        &state,
        reference.heuristic_warehouse_base_stock_level,
        reference.heuristic_retailer_base_stock_levels,
    )
    .expect("base-stock orders must compute");
    let retailer_positions = retailer_inventory_positions(&state).expect("positions must compute");
    let proportional = proportional_shipments(
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize,
        &action[1..],
    )
    .expect("proportional shipments must compute");
    let min_shortage = min_shortage_shipments(
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize,
        &action[1..],
        &retailer_positions,
        reference.heuristic_retailer_base_stock_levels,
    )
    .expect("min-shortage shipments must compute");

    assert_eq!(
        action,
        reference.expected_proportional_first_action.to_vec()
    );
    assert_eq!(proportional, reference.expected_proportional_shipments.to_vec());
    assert_eq!(
        min_shortage,
        reference.expected_min_shortage_shipments.to_vec()
    );
}

#[test]
fn finite_horizon_dp_and_heuristics_match_reference_numbers() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("optimal finite-horizon DP must solve");
    let proportional = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_proportional",
    )
    .expect("proportional heuristic evaluation must solve");
    let min_shortage = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_min_shortage",
    )
    .expect("min-shortage heuristic evaluation must solve");

    assert!(
        (optimal.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        optimal.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action.to_vec()
    );
    assert!(
        (proportional.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_proportional_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        proportional.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_proportional_first_action.to_vec()
    );
    assert!(
        (min_shortage.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_min_shortage_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        min_shortage.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_min_shortage_first_action.to_vec()
    );
}
