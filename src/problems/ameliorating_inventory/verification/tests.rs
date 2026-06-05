use crate::problems::ameliorating_inventory::env::{
    build_policy_state, initialize_state, step_state,
};
use crate::problems::ameliorating_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::ameliorating_inventory::heuristics::{
    newsvendor_purchase_order_quantity, two_dimensional_order_up_to_order_quantity,
};
use crate::problems::ameliorating_inventory::literature::{
    PAHR_GRUNOW_2025_REFERENCE, PAHR_GRUNOW_2025_REPOSITORY_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    VERIFICATION_PROBLEM_INSTANCE,
};

#[derive(Clone, Copy)]
struct WorkedTransitionCase {
    initial_inventory_by_age: &'static [usize],
    target_ages: &'static [usize],
    product_prices: &'static [f64],
    age_retention: &'static [f64],
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: &'static [f64],
    purchase_quantity: usize,
    realized_demands: &'static [usize],
    expected_shipments_by_product_age: &'static [&'static [usize]],
    expected_shipped_by_product: &'static [usize],
    expected_lost_sales_by_product: &'static [usize],
    expected_next_inventory_by_age: &'static [usize],
    expected_decayed_units_by_age: &'static [usize],
    expected_revenue: f64,
    expected_purchase_cost: f64,
    expected_holding_cost: f64,
    expected_salvage_credit: f64,
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_inventory_by_age: &[1, 2, 1],
    target_ages: &[1, 2],
    product_prices: &[5.0, 9.0],
    age_retention: &[1.0, 1.0, 1.0],
    purchase_cost_per_unit: 3.0,
    holding_cost_per_unit: 0.5,
    decay_salvage_values: &[0.0, 0.0, 0.0],
    purchase_quantity: 1,
    realized_demands: &[1, 1],
    expected_shipments_by_product_age: &[&[0, 1, 0], &[0, 0, 1]],
    expected_shipped_by_product: &[1, 1],
    expected_lost_sales_by_product: &[0, 0],
    expected_next_inventory_by_age: &[0, 2, 1],
    expected_decayed_units_by_age: &[0, 0, 0],
    expected_revenue: 14.0,
    expected_purchase_cost: 3.0,
    expected_holding_cost: 1.5,
    expected_salvage_credit: 0.0,
    expected_period_cost: -9.5,
};

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(PAHR_GRUNOW_2025_REFERENCE.benchmark_policies.len(), 4);
    assert!(PAHR_GRUNOW_2025_REFERENCE.reported_numbers_available);
    assert!(!PAHR_GRUNOW_2025_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(
        PAHR_GRUNOW_2025_REPOSITORY_REFERENCE.benchmark_policies,
        &[
            "newsvendor_purchase",
            "two_dimensional_order_up_to",
            "rolling_lp"
        ]
    );
    assert!(PAHR_GRUNOW_2025_REPOSITORY_REFERENCE.reported_numbers_available);
    assert!(!PAHR_GRUNOW_2025_REPOSITORY_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_ages, 5);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.target_ages, &[1, 3]);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.target_ages, &[1, 2]);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity, 4);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let state = initialize_state(VERIFICATION_PROBLEM_INSTANCE.initial_inventory_by_age)
        .expect("state must build");
    let features = build_policy_state(&state, &[0.8, 0.6], VERIFICATION_PROBLEM_INSTANCE.periods)
        .expect("policy state must build");

    assert_eq!(features.len(), 7);
    assert!((features[0] - 0.5).abs() < 1e-6);
    assert!((features[1] - 0.5).abs() < 1e-6);
    assert!((features[3] - 1.0).abs() < 1e-6);
    assert!((features[4] - 0.4).abs() < 1e-6);
    assert!((features[6] - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let state = initialize_state(worked.initial_inventory_by_age).expect("state must build");
    let outcome = step_state(
        &state,
        worked.purchase_quantity,
        worked.realized_demands,
        worked.target_ages,
        worked.product_prices,
        worked.age_retention,
        worked.purchase_cost_per_unit,
        worked.holding_cost_per_unit,
        worked.decay_salvage_values,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.shipments_by_product_age,
        nested_vec(worked.expected_shipments_by_product_age)
    );
    assert_eq!(
        outcome.shipped_by_product,
        worked.expected_shipped_by_product
    );
    assert_eq!(
        outcome.lost_sales_by_product,
        worked.expected_lost_sales_by_product
    );
    assert_eq!(
        outcome.next_state.inventory_by_age,
        worked.expected_next_inventory_by_age
    );
    assert_eq!(
        outcome.decayed_units_by_age,
        worked.expected_decayed_units_by_age
    );
    assert_eq!(outcome.revenue, worked.expected_revenue);
    assert_eq!(outcome.purchase_cost, worked.expected_purchase_cost);
    assert_eq!(outcome.holding_cost, worked.expected_holding_cost);
    assert_eq!(outcome.salvage_credit, worked.expected_salvage_credit);
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn heuristic_first_actions_match_named_heuristic_evaluators() {
    let state = initialize_state(VERIFICATION_PROBLEM_INSTANCE.initial_inventory_by_age)
        .expect("state must build");
    let newsvendor = newsvendor_purchase_order_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.newsvendor_total_target,
    )
    .expect("newsvendor purchase must compute");
    let two_dimensional = two_dimensional_order_up_to_order_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.two_dimensional_total_target,
        VERIFICATION_PROBLEM_INSTANCE.two_dimensional_young_target,
        VERIFICATION_PROBLEM_INSTANCE.young_age_cutoff,
    )
    .expect("two-dimensional order-up-to must compute");

    let newsvendor_eval =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "newsvendor_purchase")
            .expect("newsvendor evaluation must solve");
    let two_dimensional_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "two_dimensional_order_up_to",
    )
    .expect("two-dimensional evaluation must solve");

    assert_eq!(newsvendor, newsvendor_eval.first_action);
    assert_eq!(two_dimensional, two_dimensional_eval.first_action);
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let newsvendor =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "newsvendor_purchase")
            .expect("newsvendor evaluation must solve");
    let two_dimensional = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "two_dimensional_order_up_to",
    )
    .expect("two-dimensional evaluation must solve");

    assert!(
        optimal.discounted_cost <= newsvendor.discounted_cost + 1e-9,
        "optimal={} newsvendor={}",
        optimal.discounted_cost,
        newsvendor.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= two_dimensional.discounted_cost + 1e-9,
        "optimal={} two_dimensional={}",
        optimal.discounted_cost,
        two_dimensional.discounted_cost
    );
}
