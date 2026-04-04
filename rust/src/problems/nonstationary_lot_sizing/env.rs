use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq)]
pub struct NonstationaryLotSizingState {
    pub forecast_window: Vec<f64>,
    pub net_inventory: f64,
    pub pipeline_orders: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NonstationaryLotSizingStepOutcome {
    pub next_state: NonstationaryLotSizingState,
    pub unmet_demand: f64,
    pub holding_inventory: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(
    state: &NonstationaryLotSizingState,
    forecast_horizon: usize,
    lead_time: usize,
) -> PyResult<()> {
    if forecast_horizon < 1 {
        return Err(PyValueError::new_err("forecast_horizon must be at least 1"));
    }
    if state.forecast_window.len() != forecast_horizon {
        return Err(PyValueError::new_err(format!(
            "forecast_window length {} does not match forecast_horizon {}",
            state.forecast_window.len(),
            forecast_horizon
        )));
    }
    if state.pipeline_orders.len() != lead_time {
        return Err(PyValueError::new_err(format!(
            "pipeline_orders length {} does not match lead_time {}",
            state.pipeline_orders.len(),
            lead_time
        )));
    }
    if !state.net_inventory.is_finite() {
        return Err(PyValueError::new_err("net_inventory must be finite"));
    }
    if state
        .forecast_window
        .iter()
        .chain(state.pipeline_orders.iter())
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "forecast_window and pipeline_orders must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn initialize_state(
    forecast_window: &[f64],
    initial_net_inventory: f64,
    lead_time: usize,
) -> PyResult<NonstationaryLotSizingState> {
    let state = NonstationaryLotSizingState {
        forecast_window: forecast_window.to_vec(),
        net_inventory: initial_net_inventory,
        pipeline_orders: vec![0.0; lead_time],
    };
    validate_state(&state, forecast_window.len(), lead_time)?;
    Ok(state)
}

pub fn inventory_position(state: &NonstationaryLotSizingState) -> f64 {
    state.net_inventory + state.pipeline_orders.iter().sum::<f64>()
}

pub fn build_policy_state(state: &NonstationaryLotSizingState) -> Vec<f32> {
    let scale = (state.forecast_window.iter().copied().sum::<f64>()
        / (state.forecast_window.len().max(1) as f64))
        .max(1.0) as f32;
    let mut features = state
        .forecast_window
        .iter()
        .map(|value| *value as f32 / scale)
        .collect::<Vec<_>>();
    features.push(state.net_inventory as f32 / scale);
    features.extend(
        state
            .pipeline_orders
            .iter()
            .map(|value| *value as f32 / scale),
    );
    features
}

fn arrival_and_next_pipeline(pipeline_orders: &[f64], order_quantity: f64) -> (f64, Vec<f64>) {
    if pipeline_orders.is_empty() {
        return (order_quantity, Vec::new());
    }
    let arrival = pipeline_orders[0];
    let mut next_pipeline = Vec::with_capacity(pipeline_orders.len());
    next_pipeline.extend_from_slice(&pipeline_orders[1..]);
    next_pipeline.push(order_quantity);
    (arrival, next_pipeline)
}

pub fn step_state(
    state: &NonstationaryLotSizingState,
    order_quantity: f64,
    realized_demand: f64,
    next_forecast_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    lost_sales: bool,
) -> PyResult<NonstationaryLotSizingStepOutcome> {
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
    if !next_forecast_mean.is_finite() || next_forecast_mean < 0.0 {
        return Err(PyValueError::new_err(
            "next_forecast_mean must be finite and non-negative",
        ));
    }

    let (arrival, next_pipeline) =
        arrival_and_next_pipeline(&state.pipeline_orders, order_quantity);
    let available_inventory = state.net_inventory + arrival;
    let unconstrained_next_inventory = available_inventory - realized_demand;
    let unmet_demand = if lost_sales {
        (realized_demand - available_inventory.max(0.0)).max(0.0)
    } else {
        (-unconstrained_next_inventory).max(0.0)
    };
    let next_inventory = if lost_sales {
        unconstrained_next_inventory.max(0.0)
    } else {
        unconstrained_next_inventory
    };
    let holding_inventory = next_inventory.max(0.0);

    let period_cost = procurement_cost * order_quantity
        + if order_quantity > 0.0 {
            fixed_order_cost
        } else {
            0.0
        }
        + shortage_cost * unmet_demand
        + holding_cost * holding_inventory;

    let mut next_forecast_window = if state.forecast_window.len() > 1 {
        state.forecast_window[1..].to_vec()
    } else {
        Vec::new()
    };
    next_forecast_window.push(next_forecast_mean);

    Ok(NonstationaryLotSizingStepOutcome {
        next_state: NonstationaryLotSizingState {
            forecast_window: next_forecast_window,
            net_inventory: next_inventory,
            pipeline_orders: next_pipeline,
        },
        unmet_demand,
        holding_inventory,
        period_cost,
        reward: -period_cost,
    })
}
