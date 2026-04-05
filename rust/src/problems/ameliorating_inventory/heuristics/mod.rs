mod newsvendor_purchase;
mod two_dimensional_order_up_to;

pub use newsvendor_purchase::newsvendor_purchase_order_quantity;
pub use two_dimensional_order_up_to::two_dimensional_order_up_to_order_quantity;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::ameliorating_inventory::demand::{sample_demand, DemandModel};
use crate::problems::ameliorating_inventory::env::{
    step_state, validate_problem_spec, validate_state, AmelioratingInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn policy_purchase_quantity(
    policy_name: &str,
    params: &[f64],
    state: &AmelioratingInventoryState,
) -> PyResult<usize> {
    match policy_name {
        "newsvendor_purchase" => {
            if params.len() != 1 {
                return Err(PyValueError::new_err(
                    "newsvendor_purchase expects params [total_target_inventory]",
                ));
            }
            newsvendor_purchase_order_quantity(state, params[0].round().max(0.0) as usize)
        }
        "two_dimensional_order_up_to" => {
            if params.len() != 3 {
                return Err(PyValueError::new_err(
                    "two_dimensional_order_up_to expects params [total_target_inventory, young_target_inventory, young_age_cutoff]",
                ));
            }
            two_dimensional_order_up_to_order_quantity(
                state,
                params[0].round().max(0.0) as usize,
                params[1].round().max(0.0) as usize,
                params[2].round().max(0.0) as usize,
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
    initial_state: &AmelioratingInventoryState,
    realized_demands: &[Vec<usize>],
    target_ages: &[usize],
    product_prices: &[f64],
    age_retention: &[f64],
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: &[f64],
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state, initial_state.inventory_by_age.len())?;
    validate_problem_spec(
        initial_state.inventory_by_age.len(),
        target_ages,
        product_prices,
        age_retention,
        decay_salvage_values,
    )?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;
    for demand in realized_demands.iter() {
        let purchase_quantity = policy_purchase_quantity(policy_name, params, &state)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            demand,
            target_ages,
            product_prices,
            age_retention,
            purchase_cost_per_unit,
            holding_cost_per_unit,
            decay_salvage_values,
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
    initial_state: &AmelioratingInventoryState,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_models: &[DemandModel],
    target_ages: &[usize],
    product_prices: &[f64],
    age_retention: &[f64],
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: &[f64],
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_state(initial_state, initial_state.inventory_by_age.len())?;
    validate_problem_spec(
        initial_state.inventory_by_age.len(),
        target_ages,
        product_prices,
        age_retention,
        decay_salvage_values,
    )?;
    if demand_models.len() != target_ages.len() {
        return Err(PyValueError::new_err(
            "demand_models length must match the number of products",
        ));
    }
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);
    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut discounted_cost = 0.0;
        let mut discount = 1.0;
        for _ in 0..periods {
            let realized_demands = demand_models
                .iter()
                .map(|model| sample_demand(&mut rng, model))
                .collect::<PyResult<Vec<_>>>()?;
            let purchase_quantity = policy_purchase_quantity(policy_name, params, &state)?;
            let outcome = step_state(
                &state,
                purchase_quantity,
                &realized_demands,
                target_ages,
                product_prices,
                age_retention,
                purchase_cost_per_unit,
                holding_cost_per_unit,
                decay_salvage_values,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }
        discounted_costs.push(discounted_cost);
    }

    let mean_cost = discounted_costs.iter().sum::<f64>() / discounted_costs.len() as f64;
    let variance = discounted_costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / discounted_costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
    })
}
