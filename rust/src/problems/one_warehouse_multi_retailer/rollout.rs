use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments, random_sequential_shipments, AllocationPolicy,
};
use crate::problems::one_warehouse_multi_retailer::demand::{
    sample_demand, validate_demand_models, DemandModel,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    build_policy_state, retailer_inventory_positions, step_state, validate_state,
    CustomerBehaviorModel, OneWarehouseMultiRetailerState,
};

#[derive(Clone)]
pub struct OneWarehouseMultiRetailerRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_models: Vec<DemandModel>,
    pub allocation_policy: AllocationPolicy,
    pub retailer_target_inventory_positions: Option<Vec<usize>>,
    pub holding_cost_warehouse: f64,
    pub holding_cost_retailers: Vec<f64>,
    pub penalty_costs_retailers: Vec<f64>,
    pub customer_behavior: CustomerBehaviorModel,
    pub emergency_shipment_probability: f64,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
) -> PyResult<()> {
    validate_state(initial_state)?;
    validate_demand_models(&config.demand_models)?;
    let num_retailers = initial_state.retailer_inventory.len();
    if config.demand_models.len() != num_retailers
        || config.holding_cost_retailers.len() != num_retailers
        || config.penalty_costs_retailers.len() != num_retailers
    {
        return Err(PyValueError::new_err(
            "all retailer-wise config vectors must match the number of retailers",
        ));
    }
    if let Some(ref targets) = config.retailer_target_inventory_positions {
        if targets.len() != num_retailers {
            return Err(PyValueError::new_err(
                "retailer_target_inventory_positions length must match the number of retailers",
            ));
        }
    }
    if config.action_spec.action_dim != num_retailers + 1 {
        return Err(PyValueError::new_err(format!(
            "action_spec.action_dim {} does not match expected {}",
            config.action_spec.action_dim,
            num_retailers + 1
        )));
    }
    let expected_input_dim = 1
        + initial_state.warehouse_pipeline.len()
        + num_retailers
        + initial_state
            .retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.len())
            .sum::<usize>()
        + 2;
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected {}",
            config.input_dim, expected_input_dim
        )));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    if config.allocation_policy == AllocationPolicy::MinShortage
        && config.retailer_target_inventory_positions.is_none()
    {
        return Err(PyValueError::new_err(
            "min_shortage rollout requires retailer_target_inventory_positions",
        ));
    }
    Ok(())
}

fn action_vector(
    flat_params: &[f32],
    state: &OneWarehouseMultiRetailerState,
    config: &OneWarehouseMultiRetailerRolloutConfig,
) -> PyResult<Vec<usize>> {
    let policy_state = build_policy_state(state, config.periods)?;
    action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )
}

fn retailer_shipments<R: Rng + ?Sized>(
    rng: &mut R,
    state: &OneWarehouseMultiRetailerState,
    retailer_orders: &[usize],
    config: &OneWarehouseMultiRetailerRolloutConfig,
) -> PyResult<Vec<usize>> {
    let available_warehouse_inventory =
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
    match config.allocation_policy {
        AllocationPolicy::Proportional => {
            proportional_shipments(available_warehouse_inventory, retailer_orders)
        }
        AllocationPolicy::RandomSequential => {
            random_sequential_shipments(rng, available_warehouse_inventory, retailer_orders)
        }
        AllocationPolicy::MinShortage => min_shortage_shipments(
            available_warehouse_inventory,
            retailer_orders,
            &retailer_inventory_positions(state)?,
            config
                .retailer_target_inventory_positions
                .as_deref()
                .expect("validated above"),
        ),
    }
}

pub fn rollout(
    flat_params: &[f32],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let realized_demands = config
            .demand_models
            .iter()
            .map(|model| sample_demand(&mut rng, model))
            .collect::<PyResult<Vec<_>>>()?;
        let actions = action_vector(flat_params, &state, config)?;
        let retailer_shipments = retailer_shipments(&mut rng, &state, &actions[1..], config)?;
        let emergency_draws = if config.customer_behavior == CustomerBehaviorModel::PartialBackorder
        {
            Some(
                (0..state.retailer_inventory.len())
                    .map(|_| rng.gen_bool(config.emergency_shipment_probability))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let outcome = step_state(
            &state,
            actions[0],
            &retailer_shipments,
            &realized_demands,
            config.holding_cost_warehouse,
            &config.holding_cost_retailers,
            &config.penalty_costs_retailers,
            config.customer_behavior,
            config.emergency_shipment_probability,
            emergency_draws.as_deref(),
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
    demands: &[Vec<usize>],
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "demands length must match config.periods",
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in demands.iter() {
        if demand.len() != state.retailer_inventory.len() {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of retailers",
            ));
        }
        let actions = action_vector(flat_params, &state, config)?;
        let retailer_shipments = retailer_shipments(&mut rng, &state, &actions[1..], config)?;
        let emergency_draws = if config.customer_behavior == CustomerBehaviorModel::PartialBackorder
        {
            Some(
                (0..state.retailer_inventory.len())
                    .map(|_| rng.gen_bool(config.emergency_shipment_probability))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let outcome = step_state(
            &state,
            actions[0],
            &retailer_shipments,
            demand,
            config.holding_cost_warehouse,
            &config.holding_cost_retailers,
            &config.penalty_costs_retailers,
            config.customer_behavior,
            config.emergency_shipment_probability,
            emergency_draws.as_deref(),
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
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

pub fn build_initial_state(
    warehouse_inventory: i32,
    warehouse_pipeline: &[usize],
    retailer_inventory: &[i32],
    retailer_pipeline: &[Vec<usize>],
) -> PyResult<OneWarehouseMultiRetailerState> {
    crate::problems::one_warehouse_multi_retailer::env::initialize_state(
        warehouse_inventory,
        warehouse_pipeline,
        retailer_inventory,
        retailer_pipeline,
    )
}
