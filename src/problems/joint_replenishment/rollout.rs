use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::joint_replenishment::demand::{
    sample_demands, validate_demand_ranges, DemandRange,
};
use crate::problems::joint_replenishment::env::{
    build_raw_state, initialize_state, step_state, validate_state, JointReplenishmentState,
};

#[derive(Clone)]
pub struct JointReplenishmentRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_ranges: Vec<DemandRange>,
    pub truck_capacity: usize,
    pub minor_order_costs: Vec<f64>,
    pub major_order_cost: f64,
    pub holding_costs: Vec<f64>,
    pub shortage_costs: Vec<f64>,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(
    config: &JointReplenishmentRolloutConfig,
    initial_state: &JointReplenishmentState,
) -> PyResult<()> {
    validate_state(initial_state)?;
    let num_items = initial_state.inventory_levels.len();
    validate_demand_ranges(&config.demand_ranges)?;
    if config.demand_ranges.len() != num_items
        || config.minor_order_costs.len() != num_items
        || config.holding_costs.len() != num_items
        || config.shortage_costs.len() != num_items
    {
        return Err(PyValueError::new_err(
            "all item-wise config vectors must match the number of items",
        ));
    }
    if config.action_spec.action_dim != num_items {
        return Err(PyValueError::new_err(format!(
            "action_spec.action_dim {} does not match num_items {}",
            config.action_spec.action_dim, num_items
        )));
    }
    if config.input_dim != num_items + 2 {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected {}",
            config.input_dim,
            num_items + 2
        )));
    }
    if config.truck_capacity == 0 {
        return Err(PyValueError::new_err(
            "truck_capacity must be strictly positive",
        ));
    }
    if !config.major_order_cost.is_finite() || config.major_order_cost < 0.0 {
        return Err(PyValueError::new_err(
            "major_order_cost must be finite and non-negative",
        ));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

fn action_quantities(
    flat_params: &[f32],
    state: &JointReplenishmentState,
    config: &JointReplenishmentRolloutConfig,
) -> PyResult<Vec<usize>> {
    let policy_state = policy_state(state, config.periods)?;
    let raw_action = action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )?;
    project_to_full_truckloads(raw_action, config.truck_capacity)
}

fn project_to_full_truckloads(
    order_quantities: Vec<usize>,
    truck_capacity: usize,
) -> PyResult<Vec<usize>> {
    if truck_capacity == 0 {
        return Err(PyValueError::new_err(
            "truck_capacity must be strictly positive",
        ));
    }
    let total_order_quantity = order_quantities.iter().sum::<usize>();
    if total_order_quantity == 0 || total_order_quantity % truck_capacity == 0 {
        return Ok(order_quantities);
    }

    let lower_multiple = (total_order_quantity / truck_capacity) * truck_capacity;
    let upper_multiple = lower_multiple + truck_capacity;
    let target_quantity =
        if total_order_quantity - lower_multiple < upper_multiple - total_order_quantity {
            lower_multiple
        } else {
            upper_multiple
        };
    if target_quantity == 0 {
        return Ok(vec![0; order_quantities.len()]);
    }

    let total_as_f64 = total_order_quantity as f64;
    let mut projected = vec![0usize; order_quantities.len()];
    let mut remainders = Vec::with_capacity(order_quantities.len());
    let mut assigned = 0usize;
    for (index, quantity) in order_quantities.iter().copied().enumerate() {
        let exact_share = target_quantity as f64 * quantity as f64 / total_as_f64;
        let floor_share = exact_share.floor() as usize;
        projected[index] = floor_share;
        assigned += floor_share;
        remainders.push((index, exact_share - floor_share as f64));
    }

    remainders.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    for (index, _) in remainders
        .into_iter()
        .take(target_quantity.saturating_sub(assigned))
    {
        projected[index] += 1;
    }
    Ok(projected)
}

fn policy_state(state: &JointReplenishmentState, total_periods: usize) -> PyResult<Vec<f32>> {
    let raw_state = build_raw_state(state)?;
    let period = raw_state.last().copied().unwrap_or(0.0) as usize;
    let inventory_levels = &raw_state[..raw_state.len().saturating_sub(1)];
    let total_inventory = inventory_levels
        .iter()
        .map(|value| *value as i32)
        .sum::<i32>();
    let scale = inventory_levels
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f32, f32::max)
        .max(total_inventory.abs() as f32)
        .max(1.0);
    let mut features = inventory_levels
        .iter()
        .map(|value| *value / scale)
        .collect::<Vec<_>>();
    features.push(total_inventory as f32 / scale);
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

pub fn rollout(
    flat_params: &[f32],
    config: &JointReplenishmentRolloutConfig,
    initial_state: &JointReplenishmentState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let demands = sample_demands(&mut rng, &config.demand_ranges)?;
        let order_quantities = action_quantities(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            &order_quantities,
            &demands,
            config.truck_capacity,
            &config.minor_order_costs,
            config.major_order_cost,
            &config.holding_costs,
            &config.shortage_costs,
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &JointReplenishmentRolloutConfig,
    initial_state: &JointReplenishmentState,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    validate_config(config, initial_state)?;
    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| rollout(flat_params, config, initial_state, *seed))
        .collect()
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &JointReplenishmentRolloutConfig,
    initial_state: &JointReplenishmentState,
    demands: &[Vec<usize>],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "demands length must equal config.periods",
        ));
    }
    let num_items = initial_state.inventory_levels.len();
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in demands.iter() {
        if demand.len() != num_items {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of items",
            ));
        }
        let order_quantities = action_quantities(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            &order_quantities,
            demand,
            config.truck_capacity,
            &config.minor_order_costs,
            config.major_order_cost,
            &config.holding_costs,
            &config.shortage_costs,
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn build_initial_state(initial_inventory_levels: &[i32]) -> PyResult<JointReplenishmentState> {
    initialize_state(initial_inventory_levels)
}
