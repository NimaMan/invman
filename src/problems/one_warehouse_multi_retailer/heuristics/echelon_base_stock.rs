use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::one_warehouse_multi_retailer::env::{
    retailer_inventory_positions, warehouse_echelon_inventory_position,
    OneWarehouseMultiRetailerState,
};

pub fn echelon_base_stock_orders(
    state: &OneWarehouseMultiRetailerState,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    if retailer_base_stock_levels.len() != state.retailer_inventory.len() {
        return Err(PyValueError::new_err(
            "retailer_base_stock_levels length must match the number of retailers",
        ));
    }
    let retailer_positions = retailer_inventory_positions(state)?;
    let warehouse_position = warehouse_echelon_inventory_position(state)?;

    let mut actions = Vec::with_capacity(retailer_base_stock_levels.len() + 1);
    actions.push((warehouse_base_stock_level as i32 - warehouse_position).max(0) as usize);
    actions.extend(
        retailer_base_stock_levels
            .iter()
            .zip(retailer_positions.iter())
            .map(|(target, position)| (*target as i32 - *position).max(0) as usize),
    );
    Ok(actions)
}
