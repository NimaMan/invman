use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::spare_parts_inventory::demand::sample_failures;
use crate::problems::spare_parts_inventory::env::{
    build_policy_state, initialize_state, operational_units, step_state, validate_state,
    SparePartsInventoryState,
};

#[derive(Clone)]
pub struct SparePartsInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub installed_base: usize,
    pub failure_probability: f64,
    pub holding_cost: f64,
    pub downtime_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

pub fn build_initial_state(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: &[usize],
    repair_pipeline: &[usize],
    installed_base: usize,
) -> PyResult<SparePartsInventoryState> {
    initialize_state(
        on_hand_inventory,
        backlog,
        procurement_pipeline,
        repair_pipeline,
        installed_base,
    )
}

fn validate_config(
    config: &SparePartsInventoryRolloutConfig,
    initial_state: &SparePartsInventoryState,
) -> PyResult<()> {
    validate_state(
        initial_state,
        config.installed_base,
        initial_state.procurement_pipeline.len(),
        initial_state.repair_pipeline.len(),
    )?;
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "spare_parts_inventory rollout expects a one-dimensional action spec",
        ));
    }
    let expected_input_dim =
        initial_state.procurement_pipeline.len() + initial_state.repair_pipeline.len() + 7;
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected state size {}",
            config.input_dim, expected_input_dim
        )));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    Ok(())
}

fn action_quantity(
    flat_params: &[f32],
    state: &SparePartsInventoryState,
    config: &SparePartsInventoryRolloutConfig,
) -> PyResult<usize> {
    let policy_state = build_policy_state(
        state,
        config.installed_base,
        config.failure_probability,
        config.periods,
    )?;
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
    config: &SparePartsInventoryRolloutConfig,
    initial_state: &SparePartsInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let failures = sample_failures(
            &mut rng,
            operational_units(&state, config.installed_base)?,
            config.failure_probability,
        )?;
        let action = action_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            action,
            failures,
            config.installed_base,
            config.holding_cost,
            config.downtime_cost,
            config.procurement_cost,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &SparePartsInventoryRolloutConfig,
    initial_state: &SparePartsInventoryState,
    realized_failures: &[usize],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if realized_failures.len() != config.periods {
        return Err(PyValueError::new_err(
            "realized_failures length must match config.periods",
        ));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for failures in realized_failures.iter() {
        if *failures > operational_units(&state, config.installed_base)? {
            return Err(PyValueError::new_err(format!(
                "realized failure path value {} exceeds current operational units",
                failures
            )));
        }
        let action = action_quantity(flat_params, &state, config)?;
        let outcome = step_state(
            &state,
            action,
            *failures,
            config.installed_base,
            config.holding_cost,
            config.downtime_cost,
            config.procurement_cost,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &SparePartsInventoryRolloutConfig,
    initial_state: &SparePartsInventoryState,
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
