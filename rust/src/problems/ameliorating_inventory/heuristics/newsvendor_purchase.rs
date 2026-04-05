use pyo3::PyResult;

use crate::problems::ameliorating_inventory::env::{total_inventory, AmelioratingInventoryState};

pub fn newsvendor_purchase_order_quantity(
    state: &AmelioratingInventoryState,
    total_target_inventory: usize,
) -> PyResult<usize> {
    Ok(total_target_inventory.saturating_sub(total_inventory(state)))
}
