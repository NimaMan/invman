use crate::problems::random_yield_inventory::env::{
    build_policy_state, initialize_state, step_state,
};
use crate::problems::random_yield_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::random_yield_inventory::heuristics::{
    weighted_newsvendor_order_quantity, yield_inflated_base_stock_parameters,
};
use crate::problems::random_yield_inventory::references::{
    CHEN_2018_REFERENCE, INDERFURTH_2015_POSITIVE_LEAD_TIMES,
    INDERFURTH_2015_PROPORTIONAL_YIELD_PAIRS, INDERFURTH_2015_REFERENCE,
    LITERATURE_BENCHMARK_FAMILIES, PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
    WORKED_TRANSITION_REFERENCE, YAN_2026_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(YAN_2026_REFERENCE.benchmark_policies.len(), 5);
    assert!(!YAN_2026_REFERENCE.reported_numbers_available);
    assert!(!YAN_2026_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(
        INDERFURTH_2015_REFERENCE.benchmark_policies,
        &["linear_inflation"]
    );
    assert!(INDERFURTH_2015_REFERENCE.reported_numbers_available);
    assert!(!INDERFURTH_2015_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(
        CHEN_2018_REFERENCE.benchmark_policies,
        &["weighted_newsvendor"]
    );
    assert!(!CHEN_2018_REFERENCE.reported_numbers_available);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.lead_time, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.periods, 12);
    assert!(!PRIMARY_REFERENCE_INSTANCE.literature_verified);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.lead_time, 2);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.periods, 5);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert!(
        (VERIFICATION_PROBLEM_INSTANCE
            .demand_probabilities
            .iter()
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn literature_benchmark_catalog_tracks_match_quality() {
    assert_eq!(LITERATURE_BENCHMARK_FAMILIES.len(), 6);
    assert!(LITERATURE_BENCHMARK_FAMILIES
        .iter()
        .all(|family| !family.benchmark_policies.is_empty()));

    let yan_family = LITERATURE_BENCHMARK_FAMILIES
        .iter()
        .find(|family| family.name == "yan2026_small_scale_exact_dp_family")
        .expect("Yan small-scale family must exist");
    assert_eq!(yan_family.model_match, "exact_model_match");
    assert_eq!(yan_family.access_level, "preview_only");
    assert!(!yan_family.reported_numbers_available);
    assert_eq!(
        yan_family.repo_assertion_basis,
        "do_not_use_for_repo_assertions"
    );

    let proportional_family = LITERATURE_BENCHMARK_FAMILIES
        .iter()
        .find(|family| family.name == "inderfurth2015_positive_lt_proportional_grid")
        .expect("Inderfurth proportional family must exist");
    assert_eq!(
        proportional_family.lead_times,
        INDERFURTH_2015_POSITIVE_LEAD_TIMES
    );
    assert_eq!(
        proportional_family.yield_rate_mean_cv_pairs,
        INDERFURTH_2015_PROPORTIONAL_YIELD_PAIRS
    );
    assert!(proportional_family.reported_numbers_available);
    assert_eq!(
        proportional_family.repo_assertion_basis,
        "related_model_aggregate_only"
    );
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(3.0, &[5.0, 2.0]).expect("state must build");
    let features = build_policy_state(&state, 0.75, 12).expect("policy state must build");

    assert_eq!(features.len(), 5);
    assert!((features[0] - 0.36363637).abs() < 1e-6);
    assert!((features[1] - 1.0).abs() < 1e-6);
    assert!((features[4] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_inventory_level,
        worked.initial_pipeline_orders,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_demand,
        worked.arrival_succeeds,
        1.0,
        9.0,
        1.0,
    )
    .expect("step must succeed");

    assert_eq!(outcome.realized_arrival, worked.expected_arrival);
    assert_eq!(
        outcome.next_state.inventory_level,
        worked.expected_next_inventory_level
    );
    assert_eq!(
        outcome.next_state.pipeline_orders,
        worked.expected_next_pipeline_orders.to_vec()
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
    assert_eq!(outcome.reward, -worked.expected_period_cost);
}

#[test]
fn linear_inflation_parameters_follow_reference_formula() {
    let (target, factor) = yield_inflated_base_stock_parameters(4.0, 0.75, 2, 1.0, 9.0)
        .expect("parameters must compute");
    assert_eq!(target, 17.0);
    assert!((factor - 1.3333333333333333).abs() < 1e-12);
}

#[test]
fn weighted_newsvendor_initial_order_is_stable() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level as f64,
        &VERIFICATION_PROBLEM_INSTANCE
            .initial_pipeline_orders
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    )
    .expect("state must build");
    let order = weighted_newsvendor_order_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE
            .demand_support
            .iter()
            .zip(VERIFICATION_PROBLEM_INSTANCE.demand_probabilities.iter())
            .map(|(value, probability)| *value as f64 * probability)
            .sum(),
        VERIFICATION_PROBLEM_INSTANCE.success_probability,
        VERIFICATION_PROBLEM_INSTANCE.holding_cost,
        VERIFICATION_PROBLEM_INSTANCE.shortage_cost,
    )
    .expect("WNH order must compute");
    assert_eq!(
        order as usize,
        VERIFICATION_PROBLEM_INSTANCE.expected_weighted_newsvendor_first_action
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let linear_inflation =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "linear_inflation")
            .expect("linear inflation evaluation must solve");
    let weighted_newsvendor =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "weighted_newsvendor")
            .expect("weighted newsvendor evaluation must solve");

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
        (linear_inflation.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_linear_inflation_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        linear_inflation.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_linear_inflation_first_action
    );
    assert!(
        (weighted_newsvendor.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_weighted_newsvendor_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        weighted_newsvendor.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_weighted_newsvendor_first_action
    );
}
