// env.rs
// ======
// Executable MDP for the single-item, periodic-review, all-or-nothing random-yield inventory problem
// with positive deterministic lead time, finite horizon, discounted cost, full backlogging
// (structural match to Yan et al. 2026). Raw state only; no normalization/expectation features here.
//
// STATE: (period, inventory_level f64, pipeline_orders[L] f64). pipeline_orders[0] is the oldest
// outstanding order (placed L periods ago); a freshly placed order is appended to the back.
//
// step_state(state, order, demand, arrival_succeeds, h, b, c):  ORDER OF EVENTS per period
//   1. arrival:  realized_arrival = pipeline[0] if arrival_succeeds else 0   (all-or-nothing batch)
//   2. demand:   ending = inventory + realized_arrival - demand
//   3. cost:     period_cost = c*round(order)^+ + h*max(ending,0) + b*max(-ending,0)
//   4. shift:    next_pipeline = pipeline[1..] ++ [round(order)^+]; period += 1
//   reward = -period_cost. Orders are rounded to the nearest non-negative integer (round_order_quantity).
// An order placed now thus arrives after exactly L periods. There is NO physical order cap here
// (the DP cap lives only in finite_horizon_dp.rs for tractability).
//
// expected_inventory_position(state, p) = inventory + p * sum(pipeline): the yield-adjusted inventory
// position used by the LIR/WNH heuristics.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq)]
pub struct RandomYieldInventoryState {
    pub period: usize,
    pub inventory_level: f64,
    pub pipeline_orders: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomYieldInventoryStepOutcome {
    pub next_state: RandomYieldInventoryState,
    pub realized_arrival: f64,
    pub ending_inventory_level: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(state: &RandomYieldInventoryState, lead_time: usize) -> PyResult<()> {
    if lead_time == 0 {
        return Err(PyValueError::new_err(
            "lead_time must be at least 1 for random_yield_inventory",
        ));
    }
    if state.pipeline_orders.len() != lead_time {
        return Err(PyValueError::new_err(format!(
            "pipeline_orders length {} does not match lead_time {}",
            state.pipeline_orders.len(),
            lead_time
        )));
    }
    if !state.inventory_level.is_finite() {
        return Err(PyValueError::new_err("inventory_level must be finite"));
    }
    if state
        .pipeline_orders
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "pipeline_orders must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn initialize_state(
    initial_inventory_level: f64,
    pipeline_orders: &[f64],
) -> PyResult<RandomYieldInventoryState> {
    let state = RandomYieldInventoryState {
        period: 0,
        inventory_level: initial_inventory_level,
        pipeline_orders: pipeline_orders.to_vec(),
    };
    validate_state(&state, pipeline_orders.len())?;
    Ok(state)
}

pub fn expected_inventory_position(
    state: &RandomYieldInventoryState,
    success_probability: f64,
) -> PyResult<f64> {
    if !success_probability.is_finite() || !(0.0..=1.0).contains(&success_probability) {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    Ok(state.inventory_level + success_probability * state.pipeline_orders.iter().sum::<f64>())
}

pub fn build_raw_state(state: &RandomYieldInventoryState) -> Vec<f32> {
    let mut raw_state = Vec::with_capacity(state.pipeline_orders.len() + 2);
    raw_state.push(state.inventory_level as f32);
    raw_state.extend(state.pipeline_orders.iter().map(|value| *value as f32));
    raw_state.push(state.period as f32);
    raw_state
}

pub fn round_order_quantity(order_quantity: f64) -> f64 {
    order_quantity.max(0.0).round()
}

pub fn step_state(
    state: &RandomYieldInventoryState,
    order_quantity: f64,
    realized_demand: f64,
    arrival_succeeds: bool,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
) -> PyResult<RandomYieldInventoryStepOutcome> {
    if !order_quantity.is_finite() || order_quantity < 0.0 {
        return Err(PyValueError::new_err(
            "order_quantity must be finite and non-negative",
        ));
    }
    if !realized_demand.is_finite() || realized_demand < 0.0 {
        return Err(PyValueError::new_err(
            "realized_demand must be finite and non-negative",
        ));
    }
    if state.pipeline_orders.is_empty() {
        return Err(PyValueError::new_err(
            "random_yield_inventory requires lead_time >= 1",
        ));
    }

    let ordered_quantity = round_order_quantity(order_quantity);
    let oldest_order = state.pipeline_orders[0];
    let realized_arrival = if arrival_succeeds { oldest_order } else { 0.0 };
    let post_arrival_inventory = state.inventory_level + realized_arrival;
    let ending_inventory_level = post_arrival_inventory - realized_demand;
    let holding_inventory = ending_inventory_level.max(0.0);
    let backlog_inventory = (-ending_inventory_level).max(0.0);
    let period_cost = procurement_cost * ordered_quantity
        + holding_cost * holding_inventory
        + shortage_cost * backlog_inventory;

    let mut next_pipeline_orders = state.pipeline_orders[1..].to_vec();
    next_pipeline_orders.push(ordered_quantity);

    Ok(RandomYieldInventoryStepOutcome {
        next_state: RandomYieldInventoryState {
            period: state.period + 1,
            inventory_level: ending_inventory_level,
            pipeline_orders: next_pipeline_orders,
        },
        realized_arrival,
        ending_inventory_level,
        period_cost,
        reward: -period_cost,
    })
}
