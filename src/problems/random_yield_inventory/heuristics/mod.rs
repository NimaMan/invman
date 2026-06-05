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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscountedCostSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
    pub min_cost: f64,
    pub max_cost: f64,
    pub num_samples: usize,
}

fn summarize_costs(costs: &[f64]) -> PyResult<DiscountedCostSummary> {
    if costs.is_empty() {
        return Err(PyValueError::new_err("costs must be non-empty"));
    }
    let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance = costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / costs.len() as f64;
    let min_cost = costs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_cost = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(DiscountedCostSummary {
        mean_cost,
        cost_std: variance.sqrt(),
        min_cost,
        max_cost,
        num_samples: costs.len(),
    })
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

#[allow(clippy::too_many_arguments)]
pub fn policy_rollout(
    policy_name: &str,
    params: &[f64],
    initial_inventory_level: f64,
    pipeline_orders: &[f64],
    periods: usize,
    seed: u64,
    demand_mean: f64,
    demand_kind: DemandDistributionKind,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if !success_probability.is_finite() || !(0.0..=1.0).contains(&success_probability) {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let initial_state = initialize_state(initial_inventory_level, pipeline_orders)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state;
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

    Ok(total_cost)
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
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
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

#[allow(clippy::too_many_arguments)]
pub fn policy_discounted_cost_summary(
    policy_name: &str,
    params: &[f64],
    initial_inventory_level: f64,
    pipeline_orders: &[f64],
    periods: usize,
    seeds: &[u64],
    demand_mean: f64,
    demand_kind: DemandDistributionKind,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<DiscountedCostSummary> {
    let mut costs = Vec::with_capacity(seeds.len());
    for seed in seeds.iter().copied() {
        costs.push(policy_rollout(
            policy_name,
            params,
            initial_inventory_level,
            pipeline_orders,
            periods,
            seed,
            demand_mean,
            demand_kind,
            success_probability,
            holding_cost,
            shortage_cost,
            procurement_cost,
            discount_factor,
        )?);
    }
    summarize_costs(&costs)
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
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
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

    let summary = summarize_costs(&returns)?;

    Ok(PolicySimulationSummary {
        mean_cost: summary.mean_cost,
        cost_std: summary.cost_std,
    })
}
