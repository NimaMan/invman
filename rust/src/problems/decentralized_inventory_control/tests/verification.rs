use crate::problems::decentralized_inventory_control::env::{
    build_local_policy_state, initialize_state, step_state,
};
use crate::problems::decentralized_inventory_control::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::decentralized_inventory_control::heuristics::{
    base_stock_orders, sterman_anchor_adjust_orders,
};
use crate::problems::decentralized_inventory_control::references::{
    CANER_2014_REFERENCE, MOUSA_2024_REFERENCE, OROOJLOYJADID_2021_ALL_STERMAN_BENCHMARK,
    OROOJLOYJADID_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
    WORKED_TRANSITION_REFERENCE,
};

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(OROOJLOYJADID_2021_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(CANER_2014_REFERENCE.benchmark_policies, &["sterman_anchor_adjust"]);
    assert_eq!(
        OROOJLOYJADID_2021_ALL_STERMAN_BENCHMARK.total_mean_cost,
        45.13
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_agents, 4);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.sterman_target_positions[3], 20.0);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantities, &[4, 4]);
    assert_eq!(MOUSA_2024_REFERENCE.benchmark_policies.len(), 2);
}

#[test]
fn local_policy_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_shipment_pipelines),
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_order_pipelines),
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_shipments,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_forecast_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_actions,
    )
    .expect("state must build");
    let features = build_local_policy_state(
        &state,
        0,
        VERIFICATION_PROBLEM_INSTANCE.periods,
        VERIFICATION_PROBLEM_INSTANCE.holding_costs,
        VERIFICATION_PROBLEM_INSTANCE.backlog_costs,
    )
    .expect("local policy state must build");

    assert_eq!(features.len(), 12);
    assert!((features[0] - 1.0).abs() < 1e-6);
    assert!((features[2] - 1.0).abs() < 1e-6);
    assert!((features[6] - 0.5).abs() < 1e-6);
    assert!((features[8] - 0.5).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        &nested_vec(worked.initial_shipment_pipelines),
        &nested_vec(worked.initial_order_pipelines),
        worked.initial_last_received_shipments,
        worked.initial_last_received_orders,
        worked.initial_forecast_orders,
        worked.initial_last_actions,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_customer_demand,
        worked.demand_smoothing_factors,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("step must succeed");

    assert_eq!(outcome.received_shipments, worked.expected_received_shipments);
    assert_eq!(outcome.received_orders, worked.expected_received_orders);
    assert_eq!(outcome.downstream_shipments, worked.expected_downstream_shipments);
    assert_eq!(
        outcome.next_state.on_hand_inventory,
        worked.expected_next_on_hand_inventory
    );
    assert_eq!(outcome.next_state.backlog, worked.expected_next_backlog);
    assert_eq!(
        outcome.next_state.shipment_pipelines,
        nested_vec(worked.expected_next_shipment_pipelines)
    );
    assert_eq!(
        outcome.next_state.order_pipelines,
        nested_vec(worked.expected_next_order_pipelines)
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn heuristic_first_actions_match_reference_freeze() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_shipment_pipelines),
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_order_pipelines),
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_shipments,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_forecast_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_actions,
    )
    .expect("state must build");
    let base_stock =
        base_stock_orders(&state, VERIFICATION_PROBLEM_INSTANCE.base_stock_levels).expect("base-stock must compute");
    let sterman = sterman_anchor_adjust_orders(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.sterman_target_positions,
        VERIFICATION_PROBLEM_INSTANCE.sterman_adjustment_times,
        VERIFICATION_PROBLEM_INSTANCE.sterman_supply_line_weights,
    )
    .expect("Sterman heuristic must compute");

    assert_eq!(
        base_stock,
        VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_first_action.to_vec()
    );
    assert_eq!(
        sterman,
        VERIFICATION_PROBLEM_INSTANCE.expected_sterman_first_action.to_vec()
    );
}

#[test]
fn exact_dp_and_heuristics_match_reference_numbers() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let sterman =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "sterman_anchor_adjust")
            .expect("Sterman evaluation must solve");

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
        (base_stock.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        base_stock.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_first_action.to_vec()
    );
    assert!(
        (sterman.discounted_cost
            - VERIFICATION_PROBLEM_INSTANCE.expected_sterman_discounted_cost)
            .abs()
            < 1e-9
    );
    assert_eq!(
        sterman.first_action,
        VERIFICATION_PROBLEM_INSTANCE.expected_sterman_first_action.to_vec()
    );
}
