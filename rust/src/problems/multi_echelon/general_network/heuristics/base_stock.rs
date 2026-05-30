use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::multi_echelon::general_network::env::{
    incoming_supply_relation_indices, outgoing_edge_indices, supply_relation_count,
    supply_relations, NetworkInventoryGraph, NetworkInventoryState,
};

pub fn pairwise_base_stock_requests(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    base_stock_levels: &[usize],
    realized_external_demands: &[usize],
) -> PyResult<Vec<usize>> {
    if base_stock_levels.len() != supply_relation_count(graph) {
        return Err(PyValueError::new_err(
            "base_stock_levels must match the number of supply relations",
        ));
    }
    if realized_external_demands.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "realized_external_demands must match num_nodes",
        ));
    }

    let relations = supply_relations(graph);
    let mut requests = vec![0usize; relations.len()];
    let mut reverse_order = (0..graph.num_nodes).collect::<Vec<_>>();
    reverse_order.sort_by_key(|node_idx| *node_idx);
    reverse_order.reverse();

    for node_idx in reverse_order.into_iter() {
        let internal_successor_demand = outgoing_edge_indices(graph, node_idx)
            .iter()
            .map(|edge_idx| requests[*edge_idx])
            .sum::<usize>();
        let total_current_demand = realized_external_demands[node_idx] + internal_successor_demand;

        for relation_idx in incoming_supply_relation_indices(graph, node_idx) {
            let relation = relations[relation_idx];
            let predecessor_backlog = relation
                .internal_edge_idx
                .map(|edge_idx| state.internal_backlog_by_edge[edge_idx])
                .unwrap_or(0);
            let in_transit = state.supply_pipelines[relation_idx].iter().sum::<usize>();
            let inventory_position = state.raw_inventory_by_relation[relation_idx] as i32
                - total_current_demand as i32
                + in_transit as i32
                + predecessor_backlog as i32;
            requests[relation_idx] =
                base_stock_levels[relation_idx].saturating_sub(inventory_position.max(0) as usize);
        }
    }

    Ok(requests)
}
