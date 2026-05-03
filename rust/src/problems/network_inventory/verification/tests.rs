use crate::problems::network_inventory::env::{
    build_policy_state, initialize_state, step_state, supply_relation_count, NetworkInventoryGraph,
};
use crate::problems::network_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::network_inventory::heuristics::pairwise_base_stock_requests;
use crate::problems::network_inventory::literature::{
    PIRHOOSHYARAN_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE, SERIAL_BENCHMARK_ROWS,
    SINGLE_NODE_BENCHMARK_ROWS, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::network_inventory::verification::fixtures::WORKED_TRANSITION_CASE;
use crate::problems::network_inventory::verification::literature_benchmarks::literature_benchmark_summary;

fn build_graph(
    num_nodes: usize,
    source_nodes: &[bool],
    node_modes: &[crate::problems::network_inventory::env::NetworkNodeMode],
    external_supplier_lead_times: &[usize],
    edges: &[crate::problems::network_inventory::env::NetworkEdge],
) -> NetworkInventoryGraph {
    NetworkInventoryGraph {
        num_nodes,
        source_nodes: source_nodes.to_vec(),
        node_modes: node_modes.to_vec(),
        external_supplier_lead_times: external_supplier_lead_times.to_vec(),
        edges: edges.to_vec(),
    }
}

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(PIRHOOSHYARAN_2021_REFERENCE.benchmark_policies.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_nodes, 3);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.edges.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.pairwise_oul_levels.len(), 3);
    assert!(!PRIMARY_REFERENCE_INSTANCE.literature_verified);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.verification_source,
        "single_node_rows_verified_serial_rows_cataloged_only"
    );
    assert_eq!(SINGLE_NODE_BENCHMARK_ROWS.len(), 7);
    assert_eq!(SERIAL_BENCHMARK_ROWS.len(), 10);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_supply_requests, &[2, 2]);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn single_node_rows_match_analytical_newsvendor_values() {
    let summary = literature_benchmark_summary(32, 1234);
    assert_eq!(summary.single_node_results.len(), 7);
    for result in summary.single_node_results.iter() {
        assert!(
            (result.reproduced_analytical_oul - result.published_analytical_oul).abs() <= 0.02,
            "case {} OUL published={} reproduced={}",
            result.case_idx,
            result.published_analytical_oul,
            result.reproduced_analytical_oul
        );
        assert!(
            (result.reproduced_analytical_average_cost - result.published_analytical_average_cost)
                .abs()
                <= 0.02,
            "case {} cost published={} reproduced={}",
            result.case_idx,
            result.published_analytical_average_cost,
            result.reproduced_analytical_average_cost
        );
    }
}

#[test]
fn serial_rows_are_cataloged_but_not_verified() {
    assert_eq!(SERIAL_BENCHMARK_ROWS.len(), 10);
    assert_eq!(SERIAL_BENCHMARK_ROWS[0].published_average_cost, 22.21);
    assert_eq!(SERIAL_BENCHMARK_ROWS[2].published_average_cost, 47.65);
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.node_modes,
        VERIFICATION_PROBLEM_INSTANCE.external_supplier_lead_times,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_finished_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_raw_inventory_by_relation,
        VERIFICATION_PROBLEM_INSTANCE.initial_internal_backlog_by_edge,
        VERIFICATION_PROBLEM_INSTANCE.initial_external_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_supply_pipelines),
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
    let current_demands = vec![0usize, 1usize];
    let features = build_policy_state(
        &graph,
        &state,
        &demand_means,
        &current_demands,
        VERIFICATION_PROBLEM_INSTANCE.periods,
    )
    .expect("policy state must build");

    let expected_len =
        7 * graph.num_nodes + 2 * supply_relation_count(&graph) + graph.edges.len() + 1;
    assert_eq!(features.len(), expected_len);
    assert!((features.last().copied().unwrap_or(-1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let graph = build_graph(
        worked.num_nodes,
        worked.source_nodes,
        worked.node_modes,
        worked.external_supplier_lead_times,
        worked.edges,
    );
    let state = initialize_state(
        &graph,
        worked.initial_finished_inventory,
        worked.initial_raw_inventory_by_relation,
        worked.initial_internal_backlog_by_edge,
        worked.initial_external_backlog,
        &nested_vec(worked.initial_supply_pipelines),
    )
    .expect("state must build");
    let outcome = step_state(
        &graph,
        &state,
        worked.action,
        worked.realized_external_demands,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.received_shipments_by_relation,
        worked.expected_received_shipments_by_relation
    );
    assert_eq!(
        outcome.produced_finished_goods,
        worked.expected_produced_finished_goods
    );
    assert_eq!(
        outcome.shipped_on_internal_edges,
        worked.expected_shipped_on_internal_edges
    );
    assert_eq!(
        outcome.shipped_to_external_customer,
        worked.expected_shipped_to_external_customer
    );
    assert_eq!(
        outcome.next_state.finished_inventory,
        worked.expected_next_finished_inventory
    );
    assert_eq!(
        outcome.next_state.raw_inventory_by_relation,
        worked.expected_next_raw_inventory_by_relation
    );
    assert_eq!(
        outcome.next_state.internal_backlog_by_edge,
        worked.expected_next_internal_backlog_by_edge
    );
    assert_eq!(
        outcome.next_state.external_backlog,
        worked.expected_next_external_backlog
    );
    assert_eq!(
        outcome.next_state.supply_pipelines,
        nested_vec(worked.expected_next_supply_pipelines)
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn pairwise_base_stock_first_action_matches_reference_freeze() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.node_modes,
        VERIFICATION_PROBLEM_INSTANCE.external_supplier_lead_times,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_finished_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_raw_inventory_by_relation,
        VERIFICATION_PROBLEM_INSTANCE.initial_internal_backlog_by_edge,
        VERIFICATION_PROBLEM_INSTANCE.initial_external_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_supply_pipelines),
    )
    .expect("state must build");
    let realized_demands = vec![0usize, 1usize];
    let action = pairwise_base_stock_requests(
        &graph,
        &state,
        VERIFICATION_PROBLEM_INSTANCE.base_stock_levels,
        &realized_demands,
    )
    .expect("pairwise base-stock requests must compute");

    let base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "pairwise_base_stock")
            .expect("base-stock evaluation must solve");
    assert_eq!(action, base_stock.first_action);
}

#[test]
fn exact_dp_dominates_pairwise_base_stock() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "pairwise_base_stock")
            .expect("base-stock evaluation must solve");

    assert!(
        optimal.discounted_cost <= base_stock.discounted_cost + 1e-9,
        "optimal={} pairwise_base_stock={}",
        optimal.discounted_cost,
        base_stock.discounted_cost
    );
}
