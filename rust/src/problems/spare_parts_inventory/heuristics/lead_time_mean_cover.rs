use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::spare_parts_inventory::demand::validate_failure_probability;
use crate::problems::spare_parts_inventory::env::SparePartsInventoryState;

pub fn lead_time_mean_cover_target(
    installed_base: usize,
    failure_probability: f64,
    procurement_lead_time: usize,
    safety_buffer: f64,
) -> PyResult<usize> {
    validate_failure_probability(failure_probability)?;
    if procurement_lead_time == 0 {
        return Err(PyValueError::new_err(
            "procurement_lead_time must be at least 1",
        ));
    }
    if !safety_buffer.is_finite() || safety_buffer < 0.0 {
        return Err(PyValueError::new_err(
            "safety_buffer must be finite and non-negative",
        ));
    }
    let target =
        installed_base as f64 * failure_probability * procurement_lead_time as f64 + safety_buffer;
    Ok(target.ceil().max(0.0) as usize)
}

pub fn lead_time_mean_cover_order_quantity(
    state: &SparePartsInventoryState,
    installed_base: usize,
    failure_probability: f64,
    safety_buffer: f64,
) -> PyResult<usize> {
    let procurement_lead_time = state.procurement_pipeline.len();
    let target = lead_time_mean_cover_target(
        installed_base,
        failure_probability,
        procurement_lead_time,
        safety_buffer,
    )?;
    let near_term_cover_position = state.on_hand_inventory as i32
        + state.procurement_pipeline.iter().sum::<usize>() as i32
        + state
            .repair_pipeline
            .iter()
            .take(procurement_lead_time)
            .sum::<usize>() as i32
        - state.backlog as i32;
    Ok((target as i32 - near_term_cover_position).max(0) as usize)
}
