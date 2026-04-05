use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::vendor_managed_inventory::demand::{sample_demand, DemandDistributionKind};
use crate::problems::vendor_managed_inventory::env::{
    build_policy_state, clip_action, step_state, terminal_salvage_credit, validate_state,
    VendorManagedInventoryState,
};

#[derive(Clone)]
pub struct VendorManagedInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_kind: DemandDistributionKind,
    pub demand_mean: f64,
    pub dc_replenishment_quantity: usize,
    pub dc_capacity: usize,
    pub shipment_cost_per_unit: f64,
    pub dc_holding_cost_per_unit: f64,
    pub retailer_holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub max_shipment_quantity: usize,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(config: &VendorManagedInventoryRolloutConfig) -> PyResult<()> {
    if config.input_dim != 7 {
        return Err(PyValueError::new_err(
            "vendor_managed_inventory rollout expects input_dim = 7",
        ));
    }
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "vendor_managed_inventory rollout expects a one-dimensional action spec",
        ));
    }
    if config.periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

pub fn rollout(
    flat_params: &[f32],
    config: &VendorManagedInventoryRolloutConfig,
    initial_state: &VendorManagedInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(initial_state, config.dc_capacity)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;

    for period in 0..config.periods {
        let policy_state = build_policy_state(
            &state,
            config.demand_mean,
            config.periods,
            config.dc_capacity,
            config.dc_replenishment_quantity,
        )?;
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
        let shipment_quantity = clip_action(
            &state,
            raw_action[0],
            config.dc_capacity,
            config.max_shipment_quantity,
        )?;
        let realized_demand = sample_demand(&mut rng, config.demand_mean, config.demand_kind)?;
        let outcome = step_state(
            &state,
            shipment_quantity,
            realized_demand,
            config.dc_replenishment_quantity,
            config.dc_capacity,
            config.shipment_cost_per_unit,
            config.dc_holding_cost_per_unit,
            config.retailer_holding_cost_per_unit,
            config.stockout_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= config.discount_factor.powi(config.periods as i32)
        * terminal_salvage_credit(&state, config.dc_capacity, config.salvage_value_per_unit)?;
    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &VendorManagedInventoryRolloutConfig,
    initial_state: &VendorManagedInventoryState,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(params, seed)| rollout(params, config, initial_state, *seed))
        .collect()
}

pub fn rollout_from_demands(
    flat_params: &[f32],
    config: &VendorManagedInventoryRolloutConfig,
    initial_state: &VendorManagedInventoryState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(initial_state, config.dc_capacity)?;
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    for (period, demand) in demands.iter().copied().enumerate() {
        let policy_state = build_policy_state(
            &state,
            config.demand_mean,
            demands.len(),
            config.dc_capacity,
            config.dc_replenishment_quantity,
        )?;
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
        let shipment_quantity = clip_action(
            &state,
            raw_action[0],
            config.dc_capacity,
            config.max_shipment_quantity,
        )?;
        let outcome = step_state(
            &state,
            shipment_quantity,
            demand,
            config.dc_replenishment_quantity,
            config.dc_capacity,
            config.shipment_cost_per_unit,
            config.dc_holding_cost_per_unit,
            config.retailer_holding_cost_per_unit,
            config.stockout_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= config.discount_factor.powi(demands.len() as i32)
        * terminal_salvage_credit(&state, config.dc_capacity, config.salvage_value_per_unit)?;
    Ok(discounted_cost)
}
