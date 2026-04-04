mod lead_time_base_stock;
mod rolling_dp;
mod simple_ss;

pub use lead_time_base_stock::{lead_time_base_stock_level, lead_time_base_stock_order_quantity};
pub use rolling_dp::{
    rolling_dp_s_s_levels, rolling_dp_s_s_sequence, simulate_periodic_s_s_policy,
};
pub use simple_ss::{s_s_order_quantity, simple_s_s_levels, simple_s_s_order_quantity};

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::nonstationary_lot_sizing::demand::{sample_demand, DemandDistributionKind};
use crate::problems::nonstationary_lot_sizing::env::{
    step_state, validate_state, NonstationaryLotSizingState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
    pub shortage_rate: f64,
}

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    Ok(active_costs.iter().sum::<f64>() / active_costs.len() as f64)
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

fn policy_order_quantity(
    policy_name: &str,
    params: &[f64],
    state: &NonstationaryLotSizingState,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> PyResult<f64> {
    match policy_name {
        "s_s" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err("s_s expects params [s, S]"));
            }
            Ok(s_s_order_quantity(
                state.net_inventory + state.pipeline_orders.iter().sum::<f64>(),
                params[0],
                params[1],
            ))
        }
        "lead_time_base_stock" => Ok(lead_time_base_stock_order_quantity(
            state,
            holding_cost,
            shortage_cost,
            demand_cv,
            demand_kind,
        )),
        "simple_s_s" => Ok(simple_s_s_order_quantity(
            state,
            holding_cost,
            shortage_cost,
            fixed_order_cost,
            demand_cv,
            demand_kind,
        )),
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'"
        ))),
    }
}

pub fn policy_rollout_from_demands(
    policy_name: &str,
    params: &[f64],
    initial_state: &NonstationaryLotSizingState,
    forecast_means: &[f64],
    demands: &[f64],
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    lost_sales: bool,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    validate_state(
        initial_state,
        initial_state.forecast_window.len(),
        initial_state.pipeline_orders.len(),
    )?;
    validate_forecast_path(
        forecast_means,
        demands.len(),
        initial_state.forecast_window.len(),
    )?;
    if demands
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "demands must be finite and non-negative",
        ));
    }

    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for period in 0..demands.len() {
        let order_quantity = policy_order_quantity(
            policy_name,
            params,
            &state,
            holding_cost,
            shortage_cost,
            fixed_order_cost,
            demand_cv,
            demand_kind,
        )?;
        let next_forecast_mean = forecast_means[period + state.forecast_window.len()];
        let outcome = step_state(
            &state,
            order_quantity,
            demands[period],
            next_forecast_mean,
            holding_cost,
            shortage_cost,
            procurement_cost,
            fixed_order_cost,
            lost_sales,
        )?;
        epoch_costs.push(outcome.period_cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_state: &NonstationaryLotSizingState,
    forecast_means: &[f64],
    periods: usize,
    replications: usize,
    seed: u64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    lost_sales: bool,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> PyResult<PolicySimulationSummary> {
    validate_state(
        initial_state,
        initial_state.forecast_window.len(),
        initial_state.pipeline_orders.len(),
    )?;
    validate_forecast_path(forecast_means, periods, initial_state.forecast_window.len())?;
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut returns = Vec::with_capacity(replications);
    let mut total_shortage = 0.0;
    let mut total_demand = 0.0;

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut total_cost = 0.0;

        for period in 0..periods {
            let demand_mean = forecast_means[period];
            let realized_demand = sample_demand(&mut rng, demand_mean, demand_cv, demand_kind)?;
            let order_quantity = policy_order_quantity(
                policy_name,
                params,
                &state,
                holding_cost,
                shortage_cost,
                fixed_order_cost,
                demand_cv,
                demand_kind,
            )?;
            let next_forecast_mean = forecast_means[period + state.forecast_window.len()];
            let outcome = step_state(
                &state,
                order_quantity,
                realized_demand,
                next_forecast_mean,
                holding_cost,
                shortage_cost,
                procurement_cost,
                fixed_order_cost,
                lost_sales,
            )?;
            total_cost += outcome.period_cost;
            total_shortage += outcome.unmet_demand;
            total_demand += realized_demand;
            state = outcome.next_state;
        }

        returns.push(total_cost);
    }

    let mean_cost = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
        shortage_rate: if total_demand > 0.0 {
            total_shortage / total_demand
        } else {
            0.0
        },
    })
}
