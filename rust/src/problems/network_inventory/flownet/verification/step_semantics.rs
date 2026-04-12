#![allow(dead_code)]

use crate::problems::network_inventory::env::{
    initialize_state, step_state, NetworkInventoryGraph,
};
use crate::problems::network_inventory::heuristics::node_base_stock_requests;
use crate::problems::network_inventory::references::{
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

fn worked_graph() -> NetworkInventoryGraph {
    let worked = WORKED_TRANSITION_REFERENCE;
    NetworkInventoryGraph {
        num_nodes: worked.num_nodes,
        source_nodes: worked.source_nodes.to_vec(),
        edges: worked.edges.to_vec(),
    }
}

fn worked_state() -> crate::problems::network_inventory::env::NetworkInventoryState {
    let worked = WORKED_TRANSITION_REFERENCE;
    initialize_state(
        &worked_graph(),
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        &worked
            .initial_edge_pipelines
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )
    .expect("worked transition reference state must build")
}

pub fn verify_worked_transition_reference() -> bool {
    let worked = WORKED_TRANSITION_REFERENCE;
    let outcome = step_state(
        &worked_graph(),
        &worked_state(),
        worked.action,
        worked.realized_demands,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("worked transition must succeed");

    outcome.received_shipments_by_node == worked.expected_received_shipments_by_node
        && outcome.shipments_on_edges == worked.expected_shipments_on_edges
        && outcome.next_state.on_hand_inventory == worked.expected_next_on_hand_inventory
        && outcome.next_state.backlog == worked.expected_next_backlog
        && outcome.next_state.edge_pipelines
            == worked
                .expected_next_edge_pipelines
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>()
        && (outcome.period_cost - worked.expected_period_cost).abs() <= 1e-9
}

pub fn verify_node_base_stock_reference_action() -> bool {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let graph = NetworkInventoryGraph {
        num_nodes: reference.num_nodes,
        source_nodes: reference.source_nodes.to_vec(),
        edges: reference.edges.to_vec(),
    };
    let state = initialize_state(
        &graph,
        reference.initial_on_hand_inventory,
        reference.initial_backlog,
        &reference
            .initial_edge_pipelines
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )
    .expect("verification state must build");
    let action = node_base_stock_requests(&graph, &state, reference.base_stock_levels)
        .expect("node-base-stock action must compute");
    let base_stock = crate::problems::network_inventory::finite_horizon_dp::evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "node_base_stock",
    )
    .expect("node-base-stock evaluation must solve");

    action == base_stock.first_action
}

#[cfg(test)]
mod tests {
    use super::{verify_node_base_stock_reference_action, verify_worked_transition_reference};

    #[test]
    fn worked_transition_matches_reference_accounting() {
        assert!(verify_worked_transition_reference());
    }

    #[test]
    fn node_base_stock_first_action_matches_reference_freeze() {
        assert!(verify_node_base_stock_reference_action());
    }
}
