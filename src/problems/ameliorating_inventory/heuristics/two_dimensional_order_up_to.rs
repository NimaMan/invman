use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::ameliorating_inventory::env::{total_inventory, AmelioratingInventoryState};

pub fn young_inventory(
    state: &AmelioratingInventoryState,
    young_age_cutoff: usize,
) -> PyResult<usize> {
    if young_age_cutoff >= state.inventory_by_age.len() {
        return Err(PyValueError::new_err(format!(
            "young_age_cutoff {young_age_cutoff} is out of bounds for {} age classes",
            state.inventory_by_age.len()
        )));
    }
    Ok(state.inventory_by_age[..=young_age_cutoff].iter().sum())
}

pub fn two_dimensional_order_up_to_order_quantity(
    state: &AmelioratingInventoryState,
    total_target_inventory: usize,
    young_target_inventory: usize,
    young_age_cutoff: usize,
) -> PyResult<usize> {
    let total_gap = total_target_inventory.saturating_sub(total_inventory(state));
    let young_gap =
        young_target_inventory.saturating_sub(young_inventory(state, young_age_cutoff)?);
    Ok(total_gap.max(young_gap))
}
