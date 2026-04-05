use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::random_yield_inventory::demand::{sample_demand, DemandDistributionKind};
use crate::problems::random_yield_inventory::env::{
    build_policy_state, initialize_state, step_state, validate_state, RandomYieldInventoryState,
};

#[derive(Clone)]
pub struct RandomYieldInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_mean: f64,
    pub demand_kind: DemandDistributionKind,
    pub success_probability: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(
    config: &RandomYieldInventoryRolloutConfig,
    initial_state: &RandomYieldInventoryState,
) -> PyResult<()> {
    validate_state(initial_state, initial_state.pipeline_orders.len())?;
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "random_yield_inventory rollout expects a one-dimensional action spec",
        ));
    }
    if config.input_dim != initial_state.pipeline_orders.len() + 3 {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected state size {}",
            config.input_dim,
            initial_state.pipeline_orders.len() + 3
        )));
    }
    if !config.success_probability.is_finite() || !(0.0..=1.0).contains(&config.success_probability)
    {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

fn action_quantity(
    flat_params: &[f32],
    state: &RandomYieldInventoryState,
    config: &RandomYieldInventoryRolloutConfig,
) -> PyResult<f64> {
    let policy_state = build_policy_state(state, config.success_probability, config.periods)?;
    let action = action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )?;
    Ok(action[0] as f64)
}

pub fn rollout(
    flat_params: &[f32],
    config: &RandomYieldInventoryRolloutConfig,
    initial_state: &RandomYieldInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let realized_demand = sample_demand(&mut rng, config.demand_mean, config.demand_kind)?;
        let arrival_succeeds = rng.gen_bool(config.success_probability);
        let order_quantity = action_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            order_quantity,
            realized_demand,
            arrival_succeeds,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &RandomYieldInventoryRolloutConfig,
    initial_state: &RandomYieldInventoryState,
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
    config: &RandomYieldInventoryRolloutConfig,
    initial_state: &RandomYieldInventoryState,
    demands: &[f64],
    arrival_outcomes: &[bool],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if demands.len() != config.periods || arrival_outcomes.len() != config.periods {
        return Err(PyValueError::new_err(
            "demands and arrival_outcomes must match config.periods",
        ));
    }

    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for period in 0..config.periods {
        let order_quantity = action_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            order_quantity,
            demands[period],
            arrival_outcomes[period],
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn build_initial_state(
    initial_inventory_level: f64,
    pipeline_orders: &[f64],
) -> PyResult<RandomYieldInventoryState> {
    initialize_state(initial_inventory_level, pipeline_orders)
}
