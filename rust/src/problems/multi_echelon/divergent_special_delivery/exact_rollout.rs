use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::multi_echelon::env::{
    build_decision_state, initialize_state, step_state, AllocationMode, WarehouseBaseStockMode,
};
use crate::problems::multi_echelon::rollout::build_policy_features;

#[derive(Clone)]
pub struct MultiEchelonExactRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub discount_factor: f64,
    pub warehouse_capacity: usize,
    pub warehouse_inventory_cap: usize,
    pub retailer_inventory_cap: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub warehouse_expedited_cost: f64,
    pub warehouse_lost_sale_cost: f64,
    pub expedited_service_prob: f64,
    pub demand_support: Vec<u32>,
    pub demand_probabilities: Vec<f64>,
    pub initial_warehouse_inventory: i32,
    pub initial_warehouse_pipeline: Vec<u32>,
    pub initial_retailer_inventory: Vec<i32>,
    pub initial_retailer_pipeline: Vec<Vec<u32>>,
    pub include_period_feature: bool,
    pub warehouse_base_stock_mode: WarehouseBaseStockMode,
    pub allocation_mode: AllocationMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn sample_categorical_demand(
    rng: &mut StdRng,
    demand_support: &[u32],
    demand_probabilities: &[f64],
) -> PyResult<u32> {
    let draw = rng.gen::<f64>();
    let mut cumulative = 0.0;
    for (value, probability) in demand_support.iter().zip(demand_probabilities.iter()) {
        cumulative += *probability;
        if draw <= cumulative + 1e-12 {
            return Ok(*value);
        }
    }
    demand_support
        .last()
        .copied()
        .ok_or_else(|| PyValueError::new_err("demand_support must be non-empty"))
}

fn sample_accepted_emergency_shipments(
    rng: &mut StdRng,
    total_unmet_demand: usize,
    expedited_service_prob: f64,
) -> usize {
    (0..total_unmet_demand)
        .filter(|_| rng.gen::<f64>() < expedited_service_prob)
        .count()
}

pub fn rollout(
    flat_params: &[f32],
    config: &MultiEchelonExactRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    if config.demand_support.len() != config.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_support and demand_probabilities must have the same length",
        ));
    }
    let probability_sum = config.demand_probabilities.iter().sum::<f64>();
    if (probability_sum - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {probability_sum}"
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initialize_state(
        config.initial_warehouse_inventory,
        &config.initial_warehouse_pipeline,
        &config.initial_retailer_inventory,
        &config.initial_retailer_pipeline,
    )?;
    let mut discounted_cost = 0.0;

    for period in 0..config.periods {
        let policy_state = build_policy_features(
            &state,
            config.warehouse_inventory_cap,
            config.retailer_inventory_cap,
            config.include_period_feature,
            config.periods,
        )?;
        if policy_state.len() != config.input_dim {
            return Err(PyValueError::new_err(
                "policy state length does not match input_dim",
            ));
        }
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

        let demands = (0..config.initial_retailer_inventory.len())
            .map(|_| {
                sample_categorical_demand(
                    &mut rng,
                    &config.demand_support,
                    &config.demand_probabilities,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let decision_state = build_decision_state(&state)?;
        let total_unmet_demand = demands
            .iter()
            .enumerate()
            .map(|(retailer_idx, demand)| {
                let served =
                    (*demand).min(decision_state.retailer_available[retailer_idx].max(0) as u32);
                (*demand - served) as usize
            })
            .sum::<usize>();
        let accepted_emergency_shipments = sample_accepted_emergency_shipments(
            &mut rng,
            total_unmet_demand,
            config.expedited_service_prob,
        );

        let outcome = step_state(
            &state,
            action[0],
            action[1],
            &demands,
            accepted_emergency_shipments,
            config.warehouse_capacity,
            config.warehouse_inventory_cap,
            config.retailer_inventory_cap,
            config.warehouse_holding_cost,
            config.retailer_holding_cost,
            config.warehouse_expedited_cost,
            config.warehouse_lost_sale_cost,
            config.warehouse_base_stock_mode,
            config.allocation_mode,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &MultiEchelonExactRolloutConfig,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params batch size must match seeds size",
        ));
    }
    let results: Vec<PyResult<f64>> = params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(params, seed)| rollout(params, config, *seed))
        .collect();
    let mut values = Vec::with_capacity(results.len());
    for result in results {
        values.push(result?);
    }
    Ok(values)
}
