mod interval_stock;
mod returnability_buffer_interval_stock;

pub use interval_stock::interval_stock_action;
pub use returnability_buffer_interval_stock::returnability_buffer_interval_stock_action;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::procurement_removal_inventory::demand::{
    sample_demand, DemandDistributionKind,
};
use crate::problems::procurement_removal_inventory::env::{
    clip_action, step_state, terminal_salvage_credit, validate_state, ProcurementRemovalState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_discounted_cost: f64,
    pub std_discounted_cost: f64,
    pub min_discounted_cost: f64,
    pub max_discounted_cost: f64,
    pub num_seeds: usize,
}

fn dispatch_action(
    policy_name: &str,
    params: &[usize],
    state: &ProcurementRemovalState,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    match policy_name {
        "interval_stock" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err(
                    "interval_stock expects params [order_up_to, remove_down_to]",
                ));
            }
            interval_stock_action(
                state,
                params[0],
                params[1],
                max_purchase_quantity,
                max_removal_quantity,
            )
        }
        "returnability_buffer_interval_stock" => {
            if params.len() != 3 {
                return Err(PyValueError::new_err(
                    "returnability_buffer_interval_stock expects params [order_up_to, remove_down_to, returnable_buffer]",
                ));
            }
            returnability_buffer_interval_stock_action(
                state,
                params[0],
                params[1],
                params[2],
                max_purchase_quantity,
                max_removal_quantity,
            )
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown procurement-removal policy '{policy_name}'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn policy_rollout(
    policy_name: &str,
    params: &[usize],
    initial_state: &ProcurementRemovalState,
    periods: usize,
    seed: u64,
    demand_kind: DemandDistributionKind,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state)?;
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;

    for period in 0..periods {
        let (purchase_quantity, removal_quantity) =
            dispatch_action(policy_name, params, &state, max_purchase_quantity, max_removal_quantity)?;
        let demand = sample_demand(&mut rng, demand_mean, demand_kind)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            removal_quantity,
            demand,
            returnable_purchase_cap,
            purchase_cost_per_unit,
            return_value_per_unit,
            liquidation_value_per_unit,
            holding_cost_per_unit,
            shortage_cost_per_unit,
        )?;
        discounted_cost += discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= discount_factor.powi(periods as i32)
        * terminal_salvage_credit(&state, return_value_per_unit, liquidation_value_per_unit)?;
    Ok(discounted_cost)
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_policy(
    policy_name: &str,
    params: &[usize],
    initial_state: &ProcurementRemovalState,
    periods: usize,
    seeds: &[u64],
    demand_kind: DemandDistributionKind,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    if seeds.is_empty() {
        return Err(PyValueError::new_err("seeds must be non-empty"));
    }
    let mut costs = Vec::with_capacity(seeds.len());
    for seed in seeds.iter().copied() {
        costs.push(policy_rollout(
            policy_name,
            params,
            initial_state,
            periods,
            seed,
            demand_kind,
            demand_mean,
            returnable_purchase_cap,
            purchase_cost_per_unit,
            return_value_per_unit,
            liquidation_value_per_unit,
            holding_cost_per_unit,
            shortage_cost_per_unit,
            max_purchase_quantity,
            max_removal_quantity,
            discount_factor,
        )?);
    }
    let mean_discounted_cost = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance = costs
        .iter()
        .map(|value| {
            let centered = *value - mean_discounted_cost;
            centered * centered
        })
        .sum::<f64>()
        / costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_discounted_cost,
        std_discounted_cost: variance.sqrt(),
        min_discounted_cost: costs.iter().copied().fold(f64::INFINITY, f64::min),
        max_discounted_cost: costs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        num_seeds: costs.len(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn policy_rollout_from_demands(
    policy_name: &str,
    params: &[usize],
    initial_state: &ProcurementRemovalState,
    demands: &[usize],
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state)?;
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    for (period, demand) in demands.iter().copied().enumerate() {
        let raw_action =
            dispatch_action(policy_name, params, &state, max_purchase_quantity, max_removal_quantity)?;
        let (purchase_quantity, removal_quantity) =
            clip_action(&state, raw_action.0, raw_action.1, max_purchase_quantity, max_removal_quantity)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            removal_quantity,
            demand,
            returnable_purchase_cap,
            purchase_cost_per_unit,
            return_value_per_unit,
            liquidation_value_per_unit,
            holding_cost_per_unit,
            shortage_cost_per_unit,
        )?;
        discounted_cost += discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }
    discounted_cost -= discount_factor.powi(demands.len() as i32)
        * terminal_salvage_credit(&state, return_value_per_unit, liquidation_value_per_unit)?;
    Ok(discounted_cost)
}
