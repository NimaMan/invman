use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::nonstationary_lot_sizing::demand::{sample_demand, DemandDistributionKind};
use crate::problems::nonstationary_lot_sizing::env::{
    build_policy_state, initialize_state, step_state, validate_state, NonstationaryLotSizingState,
};

#[derive(Clone)]
pub struct NonstationaryLotSizingRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub lost_sales: bool,
    pub demand_cv: f64,
    pub demand_kind: DemandDistributionKind,
    pub warm_up_periods_ratio: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_forecast_path(
    forecast_means: &[f64],
    periods: usize,
    forecast_horizon: usize,
) -> PyResult<()> {
    let required_len = periods + forecast_horizon;
    if forecast_means.len() < required_len {
        return Err(PyValueError::new_err(format!(
            "forecast path length {} is smaller than required {}",
            forecast_means.len(),
            required_len
        )));
    }
    if forecast_means
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "forecast_means must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_config(
    config: &NonstationaryLotSizingRolloutConfig,
    initial_state: &NonstationaryLotSizingState,
) -> PyResult<()> {
    validate_state(
        initial_state,
        initial_state.forecast_window.len(),
        initial_state.pipeline_orders.len(),
    )?;
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "nonstationary lot sizing rollout expects a one-dimensional action spec",
        ));
    }
    if config.input_dim
        != initial_state.forecast_window.len() + 1 + initial_state.pipeline_orders.len()
    {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match state size {}",
            config.input_dim,
            initial_state.forecast_window.len() + 1 + initial_state.pipeline_orders.len()
        )));
    }
    if config.periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    Ok(())
}

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

fn action_quantity(
    flat_params: &[f32],
    state: &NonstationaryLotSizingState,
    config: &NonstationaryLotSizingRolloutConfig,
) -> PyResult<f64> {
    let policy_state = build_policy_state(state);
    let action = action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )?;
    Ok(action[0] as f64)
}

pub fn rollout(
    flat_params: &[f32],
    config: &NonstationaryLotSizingRolloutConfig,
    forecast_means: &[f64],
    initial_state: &NonstationaryLotSizingState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    validate_forecast_path(
        forecast_means,
        config.periods,
        initial_state.forecast_window.len(),
    )?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(config.periods);

    for period in 0..config.periods {
        let realized_demand = sample_demand(
            &mut rng,
            forecast_means[period],
            config.demand_cv,
            config.demand_kind,
        )?;
        let order_quantity = action_quantity(flat_params, &state, config)?;
        let next_forecast_mean = forecast_means[period + state.forecast_window.len()];
        let outcome = step_state(
            &state,
            order_quantity,
            realized_demand,
            next_forecast_mean,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
            config.lost_sales,
        )?;
        epoch_costs.push(outcome.period_cost);
        state = outcome.next_state;
    }

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &NonstationaryLotSizingRolloutConfig,
    forecast_means: &[f64],
    initial_state: &NonstationaryLotSizingState,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    validate_config(config, initial_state)?;
    validate_forecast_path(
        forecast_means,
        config.periods,
        initial_state.forecast_window.len(),
    )?;

    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| {
            rollout(flat_params, config, forecast_means, initial_state, *seed)
        })
        .collect()
}

pub fn rollout_from_demands(
    flat_params: &[f32],
    config: &NonstationaryLotSizingRolloutConfig,
    initial_state: &NonstationaryLotSizingState,
    forecast_means: &[f64],
    demands: &[f64],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if demands.len() != config.periods {
        return Err(PyValueError::new_err(format!(
            "demands length {} does not match config.periods {}",
            demands.len(),
            config.periods
        )));
    }
    if demands
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "demands must be finite and non-negative",
        ));
    }
    validate_forecast_path(
        forecast_means,
        config.periods,
        initial_state.forecast_window.len(),
    )?;

    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(config.periods);

    for period in 0..config.periods {
        let order_quantity = action_quantity(flat_params, &state, config)?;
        let next_forecast_mean = forecast_means[period + state.forecast_window.len()];
        let outcome = step_state(
            &state,
            order_quantity,
            demands[period],
            next_forecast_mean,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
            config.lost_sales,
        )?;
        epoch_costs.push(outcome.period_cost);
        state = outcome.next_state;
    }

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn build_initial_state_from_forecast(
    forecast_means: &[f64],
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: &[f64],
) -> PyResult<NonstationaryLotSizingState> {
    if forecast_means.len() < forecast_horizon {
        return Err(PyValueError::new_err(format!(
            "forecast_means length {} is smaller than forecast_horizon {}",
            forecast_means.len(),
            forecast_horizon
        )));
    }
    let mut state = initialize_state(
        &forecast_means[..forecast_horizon],
        initial_net_inventory,
        pipeline_orders.len(),
    )?;
    state.pipeline_orders = pipeline_orders.to_vec();
    validate_state(&state, forecast_horizon, pipeline_orders.len())?;
    Ok(state)
}
