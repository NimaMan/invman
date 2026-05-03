mod dynamic_order_up_to;
mod minimum_order_quantity;

pub use dynamic_order_up_to::dynamic_order_up_to_order_quantities;
pub use minimum_order_quantity::minimum_order_quantity_order_quantities;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::joint_replenishment::demand::{sample_demands, DemandRange};
use crate::problems::joint_replenishment::env::{
    initialize_state, step_state, validate_state, JointReplenishmentState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn parse_moq_params(params: &[f64], num_items: usize) -> PyResult<(Vec<usize>, usize, f64)> {
    if params.len() != num_items + 2 {
        return Err(PyValueError::new_err(format!(
            "minimum_order_quantity expects {} params: {} item targets, review_period, rounding_threshold",
            num_items + 2,
            num_items
        )));
    }
    let item_targets = params[..num_items]
        .iter()
        .map(|value| value.round().max(0.0) as usize)
        .collect::<Vec<_>>();
    let review_period = params[num_items].round().max(1.0) as usize;
    let rounding_threshold = params[num_items + 1].max(0.0);
    Ok((item_targets, review_period, rounding_threshold))
}

fn parse_dynout_params(params: &[f64], num_items: usize) -> PyResult<Vec<usize>> {
    if params.len() != num_items {
        return Err(PyValueError::new_err(format!(
            "dynamic_order_up_to expects {} item target params",
            num_items
        )));
    }
    Ok(params
        .iter()
        .map(|value| value.round().max(0.0) as usize)
        .collect())
}

fn policy_order_quantities(
    policy_name: &str,
    params: &[f64],
    state: &JointReplenishmentState,
    truck_capacity: usize,
    demand_ranges: &[DemandRange],
    holding_costs: &[f64],
    shortage_costs: &[f64],
) -> PyResult<Vec<usize>> {
    match policy_name {
        "minimum_order_quantity" | "moq" => {
            let (item_targets, review_period, rounding_threshold) =
                parse_moq_params(params, state.inventory_levels.len())?;
            minimum_order_quantity_order_quantities(
                state,
                &item_targets,
                review_period,
                rounding_threshold,
                truck_capacity,
            )
        }
        "dynamic_order_up_to" | "dynout" => {
            let item_targets = parse_dynout_params(params, state.inventory_levels.len())?;
            dynamic_order_up_to_order_quantities(
                state,
                &item_targets,
                truck_capacity,
                demand_ranges,
                holding_costs,
                shortage_costs,
            )
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'",
        ))),
    }
}

pub fn policy_rollout_from_paths(
    policy_name: &str,
    params: &[f64],
    initial_state: &JointReplenishmentState,
    demands: &[Vec<usize>],
    truck_capacity: usize,
    demand_ranges: &[DemandRange],
    minor_order_costs: &[f64],
    major_order_cost: f64,
    holding_costs: &[f64],
    shortage_costs: &[f64],
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(initial_state)?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    let num_items = initial_state.inventory_levels.len();
    if demand_ranges.len() != num_items {
        return Err(PyValueError::new_err(
            "demand_ranges length must match the number of items",
        ));
    }
    for demand in demands {
        if demand.len() != num_items {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of items",
            ));
        }
    }

    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;
    for demand in demands.iter() {
        let order_quantities = policy_order_quantities(
            policy_name,
            params,
            &state,
            truck_capacity,
            demand_ranges,
            holding_costs,
            shortage_costs,
        )?;
        let outcome = step_state(
            &state,
            &order_quantities,
            demand,
            truck_capacity,
            minor_order_costs,
            major_order_cost,
            holding_costs,
            shortage_costs,
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }
    Ok(total_discounted_cost)
}

pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_inventory_levels: &[i32],
    periods: usize,
    replications: usize,
    seed: u64,
    demand_ranges: &[DemandRange],
    truck_capacity: usize,
    minor_order_costs: &[f64],
    major_order_cost: f64,
    holding_costs: &[f64],
    shortage_costs: &[f64],
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let initial_state = initialize_state(initial_inventory_levels)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut total_discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let demands = sample_demands(&mut rng, demand_ranges)?;
            let order_quantities = policy_order_quantities(
                policy_name,
                params,
                &state,
                truck_capacity,
                demand_ranges,
                holding_costs,
                shortage_costs,
            )?;
            let outcome = step_state(
                &state,
                &order_quantities,
                &demands,
                truck_capacity,
                minor_order_costs,
                major_order_cost,
                holding_costs,
                shortage_costs,
            )?;
            total_discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_costs.push(total_discounted_cost);
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
