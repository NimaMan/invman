mod linear_inflation;
mod weighted_newsvendor;

pub use linear_inflation::{
    lead_time_target_stock_level, linear_inflation_order_quantity,
    yield_inflated_base_stock_order_quantity, yield_inflated_base_stock_parameters,
};
pub use weighted_newsvendor::weighted_newsvendor_order_quantity;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::problems::random_yield_inventory::demand::{sample_demand, DemandDistributionKind};
use crate::problems::random_yield_inventory::env::{
    initialize_state, step_state, validate_state, RandomYieldInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn policy_order_quantity(
    policy_name: &str,
    params: &[f64],
    state: &RandomYieldInventoryState,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    match policy_name {
        "linear_inflation" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err(
                    "linear_inflation expects params [target_stock_level, yield_inflation_factor]",
                ));
            }
            linear_inflation_order_quantity(state, success_probability, params[0], params[1])
        }
        "yield_inflated_base_stock" => yield_inflated_base_stock_order_quantity(
            state,
            demand_mean,
            success_probability,
            holding_cost,
            shortage_cost,
        ),
        "weighted_newsvendor" => weighted_newsvendor_order_quantity(
            state,
            demand_mean,
            success_probability,
            holding_cost,
            shortage_cost,
        ),
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'"
        ))),
    }
}

pub fn policy_rollout_from_paths(
    policy_name: &str,
    params: &[f64],
    initial_state: &RandomYieldInventoryState,
    demand_mean: f64,
    demands: &[f64],
    arrival_outcomes: &[bool],
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    success_probability: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state, initial_state.pipeline_orders.len())?;
    if demands.len() != arrival_outcomes.len() {
        return Err(PyValueError::new_err(
            "demands and arrival_outcomes must have the same length",
        ));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for period in 0..demands.len() {
        let order_quantity = policy_order_quantity(
            policy_name,
            params,
            &state,
            demand_mean,
            success_probability,
            holding_cost,
            shortage_cost,
        )?;
        let outcome = step_state(
            &state,
            order_quantity,
            demands[period],
            arrival_outcomes[period],
            holding_cost,
            shortage_cost,
            procurement_cost,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_inventory_level: f64,
    pipeline_orders: &[f64],
    periods: usize,
    replications: usize,
    seed: u64,
    demand_mean: f64,
    demand_kind: DemandDistributionKind,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if !success_probability.is_finite() || !(0.0..=1.0).contains(&success_probability) {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }

    let initial_state = initialize_state(initial_inventory_level, pipeline_orders)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut returns = Vec::with_capacity(replications);

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut total_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let realized_demand = sample_demand(&mut rng, demand_mean, demand_kind)?;
            let arrival_succeeds = rng.gen_bool(success_probability);
            let order_quantity = policy_order_quantity(
                policy_name,
                params,
                &state,
                demand_mean,
                success_probability,
                holding_cost,
                shortage_cost,
            )?;
            let outcome = step_state(
                &state,
                order_quantity,
                realized_demand,
                arrival_succeeds,
                holding_cost,
                shortage_cost,
                procurement_cost,
            )?;
            total_cost += discount * outcome.period_cost;
            discount *= discount_factor;
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
    })
}
