use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::spare_parts_inventory::demand::validate_failure_probability;

#[derive(Clone, Debug, PartialEq)]
pub struct SparePartsInventoryState {
    pub period: usize,
    pub on_hand_inventory: usize,
    pub backlog: usize,
    pub procurement_pipeline: Vec<usize>,
    pub repair_pipeline: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparePartsInventoryStepOutcome {
    pub next_state: SparePartsInventoryState,
    pub realized_failures: usize,
    pub procurement_arrival: usize,
    pub repair_return: usize,
    pub post_failure_on_hand_inventory: usize,
    pub post_failure_backlog: usize,
    pub restored_units: usize,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(
    state: &SparePartsInventoryState,
    installed_base: usize,
    procurement_lead_time: usize,
    repair_lead_time: usize,
) -> PyResult<()> {
    if installed_base == 0 {
        return Err(PyValueError::new_err(
            "installed_base must be at least 1",
        ));
    }
    if procurement_lead_time == 0 || repair_lead_time == 0 {
        return Err(PyValueError::new_err(
            "procurement_lead_time and repair_lead_time must be at least 1",
        ));
    }
    if state.procurement_pipeline.len() != procurement_lead_time {
        return Err(PyValueError::new_err(format!(
            "procurement_pipeline length {} does not match procurement_lead_time {}",
            state.procurement_pipeline.len(),
            procurement_lead_time
        )));
    }
    if state.repair_pipeline.len() != repair_lead_time {
        return Err(PyValueError::new_err(format!(
            "repair_pipeline length {} does not match repair_lead_time {}",
            state.repair_pipeline.len(),
            repair_lead_time
        )));
    }
    if state.backlog > installed_base {
        return Err(PyValueError::new_err(format!(
            "backlog {} cannot exceed installed_base {}",
            state.backlog, installed_base
        )));
    }
    Ok(())
}

pub fn initialize_state(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: &[usize],
    repair_pipeline: &[usize],
    installed_base: usize,
) -> PyResult<SparePartsInventoryState> {
    let state = SparePartsInventoryState {
        period: 0,
        on_hand_inventory,
        backlog,
        procurement_pipeline: procurement_pipeline.to_vec(),
        repair_pipeline: repair_pipeline.to_vec(),
    };
    validate_state(
        &state,
        installed_base,
        procurement_pipeline.len(),
        repair_pipeline.len(),
    )?;
    Ok(state)
}

pub fn operational_units(
    state: &SparePartsInventoryState,
    installed_base: usize,
) -> PyResult<usize> {
    validate_state(
        state,
        installed_base,
        state.procurement_pipeline.len(),
        state.repair_pipeline.len(),
    )?;
    Ok(installed_base - state.backlog)
}

pub fn inventory_position(state: &SparePartsInventoryState) -> i32 {
    state.on_hand_inventory as i32
        + state.procurement_pipeline.iter().sum::<usize>() as i32
        + state.repair_pipeline.iter().sum::<usize>() as i32
        - state.backlog as i32
}

pub fn build_policy_state(
    state: &SparePartsInventoryState,
    installed_base: usize,
    failure_probability: f64,
    total_periods: usize,
) -> PyResult<Vec<f32>> {
    validate_state(
        state,
        installed_base,
        state.procurement_pipeline.len(),
        state.repair_pipeline.len(),
    )?;
    validate_failure_probability(failure_probability)?;

    let inventory_position = inventory_position(state) as f64;
    let operational_units = operational_units(state, installed_base)? as f64;
    let procurement_on_order = state.procurement_pipeline.iter().sum::<usize>() as f64;
    let repair_in_process = state.repair_pipeline.iter().sum::<usize>() as f64;
    let scale = state
        .on_hand_inventory
        .max(state.backlog)
        .max(installed_base)
        .max(procurement_on_order as usize)
        .max(repair_in_process as usize)
        .max(inventory_position.abs() as usize)
        .max(1) as f32;

    let mut features =
        Vec::with_capacity(state.procurement_pipeline.len() + state.repair_pipeline.len() + 7);
    features.push(state.on_hand_inventory as f32 / scale);
    features.push(state.backlog as f32 / scale);
    features.push(inventory_position as f32 / scale);
    features.push(operational_units as f32 / scale);
    features.extend(
        state
            .procurement_pipeline
            .iter()
            .map(|value| *value as f32 / scale),
    );
    features.extend(
        state
            .repair_pipeline
            .iter()
            .map(|value| *value as f32 / scale),
    );
    features.push(installed_base as f32 / scale);
    features.push(failure_probability as f32);
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

pub fn step_state(
    state: &SparePartsInventoryState,
    order_quantity: usize,
    realized_failures: usize,
    installed_base: usize,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
) -> PyResult<SparePartsInventoryStepOutcome> {
    validate_state(
        state,
        installed_base,
        state.procurement_pipeline.len(),
        state.repair_pipeline.len(),
    )?;
    let operating_units = operational_units(state, installed_base)?;
    if realized_failures > operating_units {
        return Err(PyValueError::new_err(format!(
            "realized_failures {} cannot exceed operational_units {}",
            realized_failures, operating_units
        )));
    }

    let satisfied_failures = state.on_hand_inventory.min(realized_failures);
    let post_failure_on_hand_inventory = state.on_hand_inventory - satisfied_failures;
    let post_failure_backlog = state.backlog + realized_failures - satisfied_failures;

    let procurement_arrival = state.procurement_pipeline[0];
    let repair_return = state.repair_pipeline[0];
    let total_arrivals = procurement_arrival + repair_return;
    let restored_units = post_failure_backlog.min(total_arrivals);
    let next_backlog = post_failure_backlog - restored_units;
    let next_on_hand_inventory =
        post_failure_on_hand_inventory + total_arrivals - restored_units;

    let mut next_procurement_pipeline = state.procurement_pipeline[1..].to_vec();
    next_procurement_pipeline.push(order_quantity);
    let mut next_repair_pipeline = state.repair_pipeline[1..].to_vec();
    next_repair_pipeline.push(realized_failures);

    let period_cost = procurement_cost * order_quantity as f64
        + holding_cost * post_failure_on_hand_inventory as f64
        + downtime_cost * post_failure_backlog as f64;

    Ok(SparePartsInventoryStepOutcome {
        next_state: SparePartsInventoryState {
            period: state.period + 1,
            on_hand_inventory: next_on_hand_inventory,
            backlog: next_backlog,
            procurement_pipeline: next_procurement_pipeline,
            repair_pipeline: next_repair_pipeline,
        },
        realized_failures,
        procurement_arrival,
        repair_return,
        post_failure_on_hand_inventory,
        post_failure_backlog,
        restored_units,
        period_cost,
        reward: -period_cost,
    })
}
