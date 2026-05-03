use crate::problems::decentralized_inventory_control::env::{
    build_local_policy_state, current_received_orders, initialize_state, step_state,
};
use crate::problems::decentralized_inventory_control::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::decentralized_inventory_control::heuristics::{
    base_stock_orders, sterman_anchor_adjust_orders,
};
use crate::problems::decentralized_inventory_control::literature::references::{
    CANER_2014_REFERENCE, CLASSIC_BEER_GAME_CUSTOMER_DEMANDS, MOUSA_2024_REFERENCE,
    OROOJLOYJADID_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE, STERMAN_1989_CLASSIC_BENCHMARK,
    STERMAN_1989_REFERENCE, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::decentralized_inventory_control::verification::classic_board_game::simulate_classic_sterman_benchmark;

#[derive(Clone, Copy, Debug)]
struct WorkedTransitionCase {
    initial_on_hand_inventory: &'static [usize],
    initial_backlog: &'static [usize],
    initial_shipment_pipelines: &'static [&'static [usize]],
    initial_order_pipelines: &'static [&'static [usize]],
    initial_last_received_shipments: &'static [usize],
    initial_last_received_orders: &'static [usize],
    initial_forecast_orders: &'static [f64],
    initial_last_actions: &'static [usize],
    action: &'static [usize],
    realized_customer_demand: usize,
    demand_smoothing_factors: &'static [f64],
    holding_costs: &'static [f64],
    backlog_costs: &'static [f64],
    expected_received_shipments: &'static [usize],
    expected_received_orders: &'static [usize],
    expected_downstream_shipments: &'static [usize],
    expected_next_on_hand_inventory: &'static [usize],
    expected_next_backlog: &'static [usize],
    expected_next_shipment_pipelines: &'static [&'static [usize]],
    expected_next_order_pipelines: &'static [&'static [usize]],
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_on_hand_inventory: &[12, 12, 12, 12],
    initial_backlog: &[0, 0, 0, 0],
    initial_shipment_pipelines: &[&[4, 4], &[4, 4], &[4, 4], &[4, 4]],
    initial_order_pipelines: &[&[], &[4], &[4], &[4]],
    initial_last_received_shipments: &[4, 4, 4, 4],
    initial_last_received_orders: &[4, 4, 4, 4],
    initial_forecast_orders: &[4.0, 4.0, 4.0, 4.0],
    initial_last_actions: &[4, 4, 4, 4],
    action: &[4, 4, 4, 4],
    realized_customer_demand: 4,
    demand_smoothing_factors: &[0.0, 0.0, 0.0, 0.0],
    holding_costs: &[0.5, 0.5, 0.5, 0.5],
    backlog_costs: &[1.0, 1.0, 1.0, 1.0],
    expected_received_shipments: &[4, 4, 4, 4],
    expected_received_orders: &[4, 4, 4, 4],
    expected_downstream_shipments: &[4, 4, 4, 4],
    expected_next_on_hand_inventory: &[12, 12, 12, 12],
    expected_next_backlog: &[0, 0, 0, 0],
    expected_next_shipment_pipelines: &[&[4, 4], &[4, 4], &[4, 4], &[4, 4]],
    expected_next_order_pipelines: &[&[], &[4], &[4], &[4]],
    expected_period_cost: 24.0,
};

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(OROOJLOYJADID_2021_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(
        STERMAN_1989_REFERENCE.benchmark_policies,
        &["sterman_anchor_adjust"]
    );
    assert_eq!(
        CANER_2014_REFERENCE.benchmark_policies,
        &["sterman_anchor_adjust"]
    );
    assert_eq!(STERMAN_1989_CLASSIC_BENCHMARK.total_mean_cost, 204.0);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_agents, 4);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.benchmark_customer_demands,
        Some(CLASSIC_BEER_GAME_CUSTOMER_DEMANDS)
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.sterman_target_positions[3], 20.0);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantities, &[4, 4]);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
    assert_eq!(MOUSA_2024_REFERENCE.benchmark_policies.len(), 2);
}

#[test]
fn classic_sterman_benchmark_matches_literature() {
    let summary = simulate_classic_sterman_benchmark();

    assert_eq!(summary.per_agent_costs, [46.0, 50.0, 54.0, 54.0]);
    assert_eq!(summary.total_cost, 204.0);
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
        1,
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
    let worked = WORKED_TRANSITION_CASE;
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

    assert_eq!(
        outcome.received_shipments,
        worked.expected_received_shipments
    );
    assert_eq!(outcome.received_orders, worked.expected_received_orders);
    assert_eq!(
        outcome.downstream_shipments,
        worked.expected_downstream_shipments
    );
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
fn heuristic_first_actions_match_named_heuristic_evaluators() {
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
    let base_stock_eval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let sterman_eval =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "sterman_anchor_adjust")
            .expect("Sterman evaluation must solve");

    for demand in VERIFICATION_PROBLEM_INSTANCE
        .customer_demand_support
        .iter()
        .copied()
    {
        let observed_orders =
            current_received_orders(&state, demand as usize).expect("observed orders must build");
        let base_stock = base_stock_orders(
            &state,
            &observed_orders,
            VERIFICATION_PROBLEM_INSTANCE.base_stock_levels,
        )
        .expect("base-stock must compute");
        let sterman = sterman_anchor_adjust_orders(
            &state,
            &observed_orders,
            VERIFICATION_PROBLEM_INSTANCE.sterman_target_positions,
            VERIFICATION_PROBLEM_INSTANCE.sterman_adjustment_times,
            VERIFICATION_PROBLEM_INSTANCE.sterman_supply_line_weights,
        )
        .expect("Sterman heuristic must compute");

        let base_stock_first_action = base_stock_eval
            .first_actions_by_customer_demand
            .iter()
            .find(|(supported_demand, _)| *supported_demand == demand)
            .map(|(_, action)| action.clone())
            .expect("base-stock branch action must exist");
        let sterman_first_action = sterman_eval
            .first_actions_by_customer_demand
            .iter()
            .find(|(supported_demand, _)| *supported_demand == demand)
            .map(|(_, action)| action.clone())
            .expect("Sterman branch action must exist");

        assert_eq!(base_stock, base_stock_first_action);
        assert_eq!(sterman, sterman_first_action);
    }
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let sterman = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "sterman_anchor_adjust")
        .expect("Sterman evaluation must solve");

    assert!(
        optimal.discounted_cost <= base_stock.discounted_cost + 1e-9,
        "optimal={} base_stock={}",
        optimal.discounted_cost,
        base_stock.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= sterman.discounted_cost + 1e-9,
        "optimal={} sterman={}",
        optimal.discounted_cost,
        sterman.discounted_cost
    );
}
