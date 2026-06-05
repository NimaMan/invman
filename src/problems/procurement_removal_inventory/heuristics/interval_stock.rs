use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::procurement_removal_inventory::env::ProcurementRemovalState;

pub fn interval_stock_action(
    state: &ProcurementRemovalState,
    order_up_to: usize,
    remove_down_to: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    if order_up_to > remove_down_to {
        return Err(PyValueError::new_err(
            "order_up_to must not exceed remove_down_to",
        ));
    }
    if state.inventory_level < order_up_to {
        return Ok((
            (order_up_to - state.inventory_level).min(max_purchase_quantity),
            0,
        ));
    }
    if state.inventory_level > remove_down_to {
        return Ok((
            0,
            (state.inventory_level - remove_down_to).min(max_removal_quantity),
        ));
    }
    Ok((0, 0))
}
