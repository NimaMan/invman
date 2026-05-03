#![allow(dead_code)]

use crate::problems::network_inventory::env::{
    initialize_state, step_state, NetworkInventoryGraph,
};
use crate::problems::network_inventory::heuristics::pairwise_base_stock_requests;
use crate::problems::network_inventory::literature::VERIFICATION_PROBLEM_INSTANCE;
use crate::problems::network_inventory::verification::fixtures::WORKED_TRANSITION_CASE;

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

fn worked_graph() -> NetworkInventoryGraph {
    let worked = WORKED_TRANSITION_CASE;
    NetworkInventoryGraph {
        num_nodes: worked.num_nodes,
        source_nodes: worked.source_nodes.to_vec(),
        node_modes: worked.node_modes.to_vec(),
        external_supplier_lead_times: worked.external_supplier_lead_times.to_vec(),
        edges: worked.edges.to_vec(),
    }
}

fn worked_state() -> crate::problems::network_inventory::env::NetworkInventoryState {
    let worked = WORKED_TRANSITION_CASE;
    initialize_state(
        &worked_graph(),
        worked.initial_finished_inventory,
        worked.initial_raw_inventory_by_relation,
        worked.initial_internal_backlog_by_edge,
        worked.initial_external_backlog,
        &nested_vec(worked.initial_supply_pipelines),
    )
    .expect("worked transition reference state must build")
}

pub fn verify_worked_transition_reference() -> bool {
    let worked = WORKED_TRANSITION_CASE;
    let outcome = step_state(
        &worked_graph(),
        &worked_state(),
        worked.action,
        worked.realized_external_demands,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("worked transition must succeed");

    outcome.received_shipments_by_relation == worked.expected_received_shipments_by_relation
        && outcome.produced_finished_goods == worked.expected_produced_finished_goods
        && outcome.shipped_on_internal_edges == worked.expected_shipped_on_internal_edges
        && outcome.shipped_to_external_customer == worked.expected_shipped_to_external_customer
        && outcome.next_state.finished_inventory == worked.expected_next_finished_inventory
        && outcome.next_state.raw_inventory_by_relation
            == worked.expected_next_raw_inventory_by_relation
        && outcome.next_state.internal_backlog_by_edge
            == worked.expected_next_internal_backlog_by_edge
        && outcome.next_state.external_backlog == worked.expected_next_external_backlog
        && outcome.next_state.supply_pipelines
            == worked
                .expected_next_supply_pipelines
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>()
        && (outcome.period_cost - worked.expected_period_cost).abs() <= 1e-9
}

pub fn verify_pairwise_base_stock_reference_action() -> bool {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let graph = NetworkInventoryGraph {
        num_nodes: reference.num_nodes,
        source_nodes: reference.source_nodes.to_vec(),
        node_modes: reference.node_modes.to_vec(),
        external_supplier_lead_times: reference.external_supplier_lead_times.to_vec(),
        edges: reference.edges.to_vec(),
    };
    let state = initialize_state(
        &graph,
        reference.initial_finished_inventory,
        reference.initial_raw_inventory_by_relation,
        reference.initial_internal_backlog_by_edge,
        reference.initial_external_backlog,
        &nested_vec(reference.initial_supply_pipelines),
    )
    .expect("verification state must build");
    let realized_demands = vec![0usize, 1usize];
    let action = pairwise_base_stock_requests(
        &graph,
        &state,
        reference.base_stock_levels,
        &realized_demands,
    )
    .expect("pairwise base-stock action must compute");
    let base_stock =
        crate::problems::network_inventory::finite_horizon_dp::evaluate_named_heuristic(
            &VERIFICATION_PROBLEM_INSTANCE,
            "pairwise_base_stock",
        )
        .expect("pairwise base-stock evaluation must solve");

    action == base_stock.first_action
}

#[cfg(test)]
mod tests {
    use super::{verify_pairwise_base_stock_reference_action, verify_worked_transition_reference};

    #[test]
    fn worked_transition_matches_reference_accounting() {
        assert!(verify_worked_transition_reference());
    }

    #[test]
    fn pairwise_base_stock_first_action_matches_reference_freeze() {
        assert!(verify_pairwise_base_stock_reference_action());
    }
}
