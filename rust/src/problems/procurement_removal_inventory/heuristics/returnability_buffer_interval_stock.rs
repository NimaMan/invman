use pyo3::PyResult;

use crate::problems::procurement_removal_inventory::env::ProcurementRemovalState;
use crate::problems::procurement_removal_inventory::heuristics::interval_stock_action;

pub fn returnability_buffer_interval_stock_action(
    state: &ProcurementRemovalState,
    order_up_to: usize,
    remove_down_to: usize,
    returnable_buffer: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    let bonus = state.returnable_inventory.min(returnable_buffer);
    interval_stock_action(
        state,
        order_up_to + bonus,
        remove_down_to + bonus,
        max_purchase_quantity,
        max_removal_quantity,
    )
}
