use pyo3::PyResult;

use crate::problems::spare_parts_inventory::env::{
    inventory_position, SparePartsInventoryState,
};

pub fn base_stock_order_quantity(
    state: &SparePartsInventoryState,
    base_stock_level: usize,
) -> PyResult<usize> {
    let gap = base_stock_level as i32 - inventory_position(state);
    Ok(gap.max(0) as usize)
}
