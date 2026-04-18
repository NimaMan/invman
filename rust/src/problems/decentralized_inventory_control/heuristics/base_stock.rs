use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::decentralized_inventory_control::env::{
    inventory_positions, DecentralizedInventoryControlState,
};

pub fn base_stock_orders(
    state: &DecentralizedInventoryControlState,
    current_received_orders: &[usize],
    base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    let inventory_positions = inventory_positions(state)?;
    if base_stock_levels.len() != inventory_positions.len()
        || current_received_orders.len() != inventory_positions.len()
    {
        return Err(PyValueError::new_err(
            "base_stock_levels and current_received_orders must match the number of agents",
        ));
    }
    Ok(base_stock_levels
        .iter()
        .zip(inventory_positions.iter())
        .zip(current_received_orders.iter())
        .map(|((target, position), observed_order)| {
            let position_after_observed_order = *position - *observed_order as i32;
            target.saturating_sub(position_after_observed_order.max(0) as usize)
        })
        .collect())
}
