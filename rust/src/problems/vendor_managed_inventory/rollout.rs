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
    build_paper_policy_state, build_policy_state, clip_action, initialize_paper_state,
    step_paper_state, step_state, terminal_salvage_credit, validate_state,
    VendorManagedInventoryState,
};
use crate::problems::vendor_managed_inventory::heuristics::{
    paper_allocate_with_trucks, paper_newsvendor_order_up_to_levels,
};
use crate::problems::vendor_managed_inventory::literature::references::build_giannoccaro_2010_case;

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

#[derive(Clone)]
pub struct VendorManagedInventoryPaperRolloutConfig {
    pub case_id: usize,
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub warmup_time: f64,
    pub evaluation_time: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_paper_config(config: &VendorManagedInventoryPaperRolloutConfig) -> PyResult<()> {
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "vendor_managed_inventory paper rollout expects a one-dimensional action spec",
        ));
    }
    if !config.warmup_time.is_finite() || config.warmup_time < 0.0 {
        return Err(PyValueError::new_err(
            "warmup_time must be finite and non-negative",
        ));
    }
    if !config.evaluation_time.is_finite() || config.evaluation_time <= 0.0 {
        return Err(PyValueError::new_err(
            "evaluation_time must be finite and strictly positive",
        ));
    }
    Ok(())
}

pub fn paper_rollout(
    flat_params: &[f32],
    config: &VendorManagedInventoryPaperRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate_paper_config(config)?;
    let model = build_giannoccaro_2010_case(config.case_id).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown Giannoccaro 2010 case_id {}",
            config.case_id
        ))
    })?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initialize_paper_state(&model, &mut rng)?;
    let probe_state = build_paper_policy_state(&model, &state)?;
    if probe_state.len() != config.input_dim {
        return Err(PyValueError::new_err(format!(
            "vendor_managed_inventory paper rollout expects input_dim = {}, found {}",
            probe_state.len(),
            config.input_dim
        )));
    }

    let mut elapsed = 0.0;
    let mut measured_time = 0.0;
    let mut measured_profit = 0.0;
    while measured_time < config.evaluation_time {
        let policy_state = build_paper_policy_state(&model, &state)?;
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
        let trucks_dispatched = raw_action[0].min(model.max_trucks);
        let order_up_to_levels = paper_newsvendor_order_up_to_levels(&model, &state)?;
        let dispatch_quantities =
            paper_allocate_with_trucks(&model, &state, &order_up_to_levels, trucks_dispatched)?;
        let outcome = step_paper_state(
            &model,
            &state,
            trucks_dispatched,
            &dispatch_quantities,
            &mut rng,
        )?;
        if elapsed >= config.warmup_time {
            measured_profit += outcome.cycle_profit;
            measured_time += outcome.route_cycle_time;
        }
        elapsed += outcome.route_cycle_time;
        state = outcome.next_state;
    }

    Ok(-(measured_profit / measured_time.max(1e-9)))
}

pub fn paper_population_rollout(
    params_batch: &[Vec<f32>],
    config: &VendorManagedInventoryPaperRolloutConfig,
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
        .map(|(params, seed)| paper_rollout(params, config, *seed))
        .collect()
}
