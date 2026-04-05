use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::ameliorating_inventory::demand::{mean_demand, sample_demand, DemandModel};
use crate::problems::ameliorating_inventory::env::{
    build_policy_state, initialize_state, step_state, validate_problem_spec, validate_state,
    AmelioratingInventoryState,
};

#[derive(Clone)]
pub struct AmelioratingInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_models: Vec<DemandModel>,
    pub target_ages: Vec<usize>,
    pub product_prices: Vec<f64>,
    pub age_retention: Vec<f64>,
    pub purchase_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub decay_salvage_values: Vec<f64>,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

pub fn build_initial_state(inventory_by_age: &[usize]) -> PyResult<AmelioratingInventoryState> {
    initialize_state(inventory_by_age)
}

fn validate_config(
    config: &AmelioratingInventoryRolloutConfig,
    initial_state: &AmelioratingInventoryState,
) -> PyResult<()> {
    validate_state(initial_state, initial_state.inventory_by_age.len())?;
    validate_problem_spec(
        initial_state.inventory_by_age.len(),
        &config.target_ages,
        &config.product_prices,
        &config.age_retention,
        &config.decay_salvage_values,
    )?;
    if config.demand_models.len() != config.target_ages.len() {
        return Err(PyValueError::new_err(
            "demand_models length must match the number of products",
        ));
    }
    let expected_input_dim =
        initial_state.inventory_by_age.len() + config.demand_models.len() + 2;
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected state size {}",
            config.input_dim, expected_input_dim
        )));
    }
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "ameliorating_inventory rollout expects a one-dimensional action spec",
        ));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    Ok(())
}

fn purchase_quantity(
    flat_params: &[f32],
    state: &AmelioratingInventoryState,
    config: &AmelioratingInventoryRolloutConfig,
) -> PyResult<usize> {
    let expected_demands = config
        .demand_models
        .iter()
        .map(mean_demand)
        .collect::<PyResult<Vec<_>>>()?;
    let policy_state = build_policy_state(state, &expected_demands, config.periods)?;
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
    Ok(action[0])
}

pub fn rollout(
    flat_params: &[f32],
    config: &AmelioratingInventoryRolloutConfig,
    initial_state: &AmelioratingInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;
    for _ in 0..config.periods {
        let realized_demands = config
            .demand_models
            .iter()
            .map(|model| sample_demand(&mut rng, model))
            .collect::<PyResult<Vec<_>>>()?;
        let purchase_quantity = purchase_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            &realized_demands,
            &config.target_ages,
            &config.product_prices,
            &config.age_retention,
            config.purchase_cost_per_unit,
            config.holding_cost_per_unit,
            &config.decay_salvage_values,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }
    Ok(discounted_cost)
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &AmelioratingInventoryRolloutConfig,
    initial_state: &AmelioratingInventoryState,
    realized_demands: &[Vec<usize>],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if realized_demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "realized_demands length must match config.periods",
        ));
    }
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;
    for demand in realized_demands.iter() {
        if demand.len() != config.target_ages.len() {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of products",
            ));
        }
        let purchase_quantity = purchase_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            demand,
            &config.target_ages,
            &config.product_prices,
            &config.age_retention,
            config.purchase_cost_per_unit,
            config.holding_cost_per_unit,
            &config.decay_salvage_values,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }
    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &AmelioratingInventoryRolloutConfig,
    initial_state: &AmelioratingInventoryState,
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
        .map(|(params, seed)| rollout(params, config, initial_state, *seed))
        .collect()
}
