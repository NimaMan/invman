use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::decentralized_inventory_control::env::{
    inventory_positions, DecentralizedInventoryControlState,
};

pub fn base_stock_orders(
    state: &DecentralizedInventoryControlState,
    base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    let inventory_positions = inventory_positions(state)?;
    if base_stock_levels.len() != inventory_positions.len() {
        return Err(PyValueError::new_err(
            "base_stock_levels must match the number of agents",
        ));
    }
    Ok(base_stock_levels
        .iter()
        .zip(inventory_positions.iter())
        .map(|(target, position)| target.saturating_sub((*position).max(0) as usize))
        .collect())
}
