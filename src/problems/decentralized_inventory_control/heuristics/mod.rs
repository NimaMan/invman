#![allow(dead_code)]

mod base_stock;
mod sterman_anchor_adjust;

pub use base_stock::base_stock_orders;
pub use sterman_anchor_adjust::sterman_anchor_adjust_orders;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::decentralized_inventory_control::demand::{sample_demand, DemandModel};
use crate::problems::decentralized_inventory_control::env::{
    current_received_orders, step_state, validate_state, DecentralizedInventoryControlState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn base_stock_levels_from_params(params: &[f64], num_agents: usize) -> PyResult<Vec<usize>> {
    if params.len() != num_agents {
        return Err(PyValueError::new_err(format!(
            "base_stock expects {} parameters",
            num_agents
        )));
    }
    Ok(params
        .iter()
        .map(|value| value.round().max(0.0) as usize)
        .collect())
}

fn sterman_params_from_flat(
    params: &[f64],
    num_agents: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if params.len() != 3 * num_agents {
        return Err(PyValueError::new_err(format!(
            "sterman_anchor_adjust expects {} parameters: target positions, adjustment times, and supply-line weights",
            3 * num_agents
        )));
    }
    Ok((
        params[..num_agents].to_vec(),
        params[num_agents..2 * num_agents].to_vec(),
        params[2 * num_agents..].to_vec(),
    ))
}

fn policy_actions(
    policy_name: &str,
    params: &[f64],
    state: &DecentralizedInventoryControlState,
    realized_customer_demand: usize,
) -> PyResult<Vec<usize>> {
    let observed_orders = current_received_orders(state, realized_customer_demand)?;
    match policy_name {
        "base_stock" => base_stock_orders(
            state,
            &observed_orders,
            &base_stock_levels_from_params(params, state.on_hand_inventory.len())?,
        ),
        "sterman_anchor_adjust" => {
            let (target_positions, adjustment_times, supply_line_weights) =
                sterman_params_from_flat(params, state.on_hand_inventory.len())?;
            sterman_anchor_adjust_orders(
                state,
                &observed_orders,
                &target_positions,
                &adjustment_times,
                &supply_line_weights,
            )
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'"
        ))),
    }
}

pub fn policy_rollout_from_paths(
    policy_name: &str,
    params: &[f64],
    initial_state: &DecentralizedInventoryControlState,
    customer_demands: &[usize],
    demand_smoothing_factors: &[f64],
    holding_costs: &[f64],
    backlog_costs: &[f64],
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state)?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in customer_demands.iter().copied() {
        let actions = policy_actions(policy_name, params, &state, demand)?;
        let outcome = step_state(
            &state,
            &actions,
            demand,
            demand_smoothing_factors,
            holding_costs,
            backlog_costs,
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
    initial_state: &DecentralizedInventoryControlState,
    periods: usize,
    replications: usize,
    seed: u64,
    customer_demand_model: &DemandModel,
    demand_smoothing_factors: &[f64],
    holding_costs: &[f64],
    backlog_costs: &[f64],
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_state(initial_state)?;
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let customer_demand = sample_demand(&mut rng, customer_demand_model)?;
            let actions = policy_actions(policy_name, params, &state, customer_demand)?;
            let outcome = step_state(
                &state,
                &actions,
                customer_demand,
                demand_smoothing_factors,
                holding_costs,
                backlog_costs,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_costs.push(discounted_cost);
    }

    let mean_cost = discounted_costs.iter().sum::<f64>() / discounted_costs.len() as f64;
    let cost_variance = discounted_costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / discounted_costs.len() as f64;

    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: cost_variance.sqrt(),
    })
}
