use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::network_inventory::env::{
    incoming_edge_indices, inventory_positions, NetworkInventoryGraph, NetworkInventoryState,
};

pub fn node_base_stock_requests(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    let inventory_positions = inventory_positions(graph, state)?;
    if base_stock_levels.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "base_stock_levels must match num_nodes",
        ));
    }

    let mut edge_requests = vec![0usize; graph.edges.len()];
    for node_idx in 0..graph.num_nodes {
        let incoming = incoming_edge_indices(graph, node_idx);
        if incoming.is_empty() {
            continue;
        }
        let gap = base_stock_levels[node_idx]
            .saturating_sub(inventory_positions[node_idx].max(0) as usize);
        if gap == 0 {
            continue;
        }
        let base_share = gap / incoming.len();
        let remainder = gap % incoming.len();
        for (offset, edge_idx) in incoming.into_iter().enumerate() {
            edge_requests[edge_idx] = base_share + usize::from(offset < remainder);
        }
    }
    Ok(edge_requests)
}
