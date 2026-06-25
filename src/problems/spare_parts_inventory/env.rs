// =============================================================================
// spare_parts_inventory::env
//
// PURPOSE
//   The trainable single-echelon PERIODIC-REVIEW repairable spare-parts MDP.
//
// MODEL (repo-native; NOT literature-verified)
//   State per period:
//     on_hand_inventory, backlog,
//     procurement_pipeline (length = procurement_lead_time),
//     repair_pipeline      (length = repair_lead_time).
//   operational_units = installed_base - backlog.
//   Per-period transition (step_state), ORDER-AFTER-DEMAND:
//     1. realized_failures occur among operational_units (binomial in demand.rs).
//     2. failures are met from on-hand first; the shortfall increases backlog.
//        post_failure_on_hand = on_hand - min(on_hand, failures)
//        post_failure_backlog = backlog + failures - min(on_hand, failures)
//     3. arrivals = procurement_pipeline[0] + repair_pipeline[0] clear backlog
//        first, then add to on-hand.
//     4. the order_quantity enters the tail of the procurement pipeline; the
//        failed units enter the tail of the repair pipeline and return
//        DETERMINISTICALLY exactly repair_lead_time periods later.
//     5. period_cost = procurement_cost * order_quantity
//                    + holding_cost   * post_failure_on_hand
//                    + downtime_cost  * post_failure_backlog.
//
// VERIFICATION STATUS (honest, per docs/rust/README.md)
//   NOT literature-verified. No paper publishes a numeric cost for this exact
//   construction (binomial failures + deterministic fixed-lead-time repair return
//   + finite-horizon DP). The Kranenburg (2006) Table 5.2 reproduction belongs to
//   the analytical CONTINUOUS-REVIEW lateral-transshipment module
//   (literature/kranenburg_lateral_transshipment.rs), which is a STRUCTURALLY
//   DIFFERENT model and does NOT verify this environment. env.rs is exercised only
//   by characterization / drift-guard tests and a self-consistency DP comparison.
//   references.rs flags PRIMARY_REFERENCE_INSTANCE and VERIFICATION_PROBLEM_INSTANCE
//   with literature_verified = false.
// =============================================================================

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

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
        return Err(PyValueError::new_err("installed_base must be at least 1"));
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

pub fn build_raw_state(state: &SparePartsInventoryState) -> PyResult<Vec<f32>> {
    let mut raw_state =
        Vec::with_capacity(state.procurement_pipeline.len() + state.repair_pipeline.len() + 3);
    raw_state.push(state.on_hand_inventory as f32);
    raw_state.push(state.backlog as f32);
    raw_state.extend(state.procurement_pipeline.iter().map(|value| *value as f32));
    raw_state.extend(state.repair_pipeline.iter().map(|value| *value as f32));
    raw_state.push(state.period as f32);
    Ok(raw_state)
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
    let next_on_hand_inventory = post_failure_on_hand_inventory + total_arrivals - restored_units;

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
