mod base_stock;
mod lead_time_mean_cover;

pub use base_stock::base_stock_order_quantity;
pub use lead_time_mean_cover::{lead_time_mean_cover_order_quantity, lead_time_mean_cover_target};

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::spare_parts_inventory::demand::sample_failures;
use crate::problems::spare_parts_inventory::env::{
    operational_units, step_state, validate_state, SparePartsInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn policy_order_quantity(
    policy_name: &str,
    params: &[f64],
    state: &SparePartsInventoryState,
    installed_base: usize,
    failure_probability: f64,
) -> PyResult<usize> {
    match policy_name {
        "base_stock" => {
            if params.len() != 1 {
                return Err(PyValueError::new_err(
                    "base_stock expects params [base_stock_level]",
                ));
            }
            base_stock_order_quantity(state, params[0].round().max(0.0) as usize)
        }
        "lead_time_mean_cover" => {
            if params.len() != 1 {
                return Err(PyValueError::new_err(
                    "lead_time_mean_cover expects params [safety_buffer]",
                ));
            }
            lead_time_mean_cover_order_quantity(
                state,
                installed_base,
                failure_probability,
                params[0],
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
    initial_state: &SparePartsInventoryState,
    installed_base: usize,
    realized_failures: &[usize],
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    failure_probability: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(
        initial_state,
        installed_base,
        initial_state.procurement_pipeline.len(),
        initial_state.repair_pipeline.len(),
    )?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for failures in realized_failures.iter() {
        let operating_units = operational_units(&state, installed_base)?;
        if *failures > operating_units {
            return Err(PyValueError::new_err(format!(
                "realized failure path value {} exceeds operational units {}",
                failures, operating_units
            )));
        }
        let order_quantity = policy_order_quantity(
            policy_name,
            params,
            &state,
            installed_base,
            failure_probability,
        )?;
        let outcome = step_state(
            &state,
            order_quantity,
            *failures,
            installed_base,
            holding_cost,
            downtime_cost,
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
    initial_state: &SparePartsInventoryState,
    periods: usize,
    replications: usize,
    seed: u64,
    installed_base: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_state(
        initial_state,
        installed_base,
        initial_state.procurement_pipeline.len(),
        initial_state.repair_pipeline.len(),
    )?;
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
    let mut costs = Vec::with_capacity(replications);
    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let failures = sample_failures(
                &mut rng,
                operational_units(&state, installed_base)?,
                failure_probability,
            )?;
            let order_quantity = policy_order_quantity(
                policy_name,
                params,
                &state,
                installed_base,
                failure_probability,
            )?;
            let outcome = step_state(
                &state,
                order_quantity,
                failures,
                installed_base,
                holding_cost,
                downtime_cost,
                procurement_cost,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        costs.push(discounted_cost);
    }

    let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance = costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
    })
}
