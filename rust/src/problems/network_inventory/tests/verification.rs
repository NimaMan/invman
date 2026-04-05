use crate::problems::network_inventory::env::{
    build_policy_state, initialize_state, step_state, NetworkInventoryGraph,
};
use crate::problems::network_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::network_inventory::heuristics::node_base_stock_requests;
use crate::problems::network_inventory::references::{
    PIRHOOSHYARAN_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
    WORKED_TRANSITION_REFERENCE,
};

fn build_graph(num_nodes: usize, source_nodes: &[bool], edges: &[crate::problems::network_inventory::env::NetworkEdge]) -> NetworkInventoryGraph {
    NetworkInventoryGraph {
        num_nodes,
        source_nodes: source_nodes.to_vec(),
        edges: edges.to_vec(),
    }
}

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(PIRHOOSHYARAN_2021_REFERENCE.benchmark_policies.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_nodes, 4);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.edges.len(), 4);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_edge_requests, &[2, 2, 2, 2]);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_edge_pipelines),
    )
    .expect("state must build");
    let demand_means = VERIFICATION_PROBLEM_INSTANCE
        .demand_supports
        .iter()
        .zip(VERIFICATION_PROBLEM_INSTANCE.demand_probabilities.iter())
        .map(|(support, probabilities)| {
            support
                .iter()
                .zip(probabilities.iter())
                .map(|(value, probability)| *value as f64 * probability)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let features = build_policy_state(
        &graph,
        &state,
        &demand_means,
        VERIFICATION_PROBLEM_INSTANCE.periods,
    )
    .expect("policy state must build");

    assert_eq!(features.len(), 4 * graph.num_nodes + graph.edges.len() + 1);
    assert!((features[1] - 0.0).abs() < 1e-6);
    assert!((features[4] - 1.0).abs() < 1e-6);
    assert!((features.last().copied().unwrap_or(-1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let graph = build_graph(worked.num_nodes, worked.source_nodes, worked.edges);
    let state = initialize_state(
        &graph,
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        &nested_vec(worked.initial_edge_pipelines),
    )
    .expect("state must build");
    let outcome = step_state(
        &graph,
        &state,
        worked.action,
        worked.realized_demands,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.received_shipments_by_node,
        worked.expected_received_shipments_by_node
    );
    assert_eq!(outcome.shipments_on_edges, worked.expected_shipments_on_edges);
    assert_eq!(
        outcome.next_state.on_hand_inventory,
        worked.expected_next_on_hand_inventory
    );
    assert_eq!(outcome.next_state.backlog, worked.expected_next_backlog);
    assert_eq!(
        outcome.next_state.edge_pipelines,
        nested_vec(worked.expected_next_edge_pipelines)
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn node_base_stock_first_action_matches_reference_freeze() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_edge_pipelines),
    )
    .expect("state must build");
    let action = node_base_stock_requests(
        &graph,
        &state,
        VERIFICATION_PROBLEM_INSTANCE.base_stock_levels,
    )
    .expect("base-stock requests must compute");

    assert_eq!(
        action,
        VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_first_action.to_vec()
    );
}

#[test]
fn exact_dp_and_base_stock_match_reference_numbers() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "node_base_stock")
            .expect("base-stock evaluation must solve");

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
}
