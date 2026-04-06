use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::procurement_removal_inventory::demand::{
    sample_demand, DemandDistributionKind,
};
use crate::problems::procurement_removal_inventory::env::{
    build_raw_state, clip_action, step_state, terminal_salvage_credit, validate_state,
    ProcurementRemovalState,
};

#[derive(Clone)]
pub struct ProcurementRemovalRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_kind: DemandDistributionKind,
    pub demand_mean: f64,
    pub returnable_purchase_cap: usize,
    pub purchase_cost_per_unit: f64,
    pub return_value_per_unit: f64,
    pub liquidation_value_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub shortage_cost_per_unit: f64,
    pub max_purchase_quantity: usize,
    pub max_removal_quantity: usize,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(config: &ProcurementRemovalRolloutConfig) -> PyResult<()> {
    if config.input_dim != 7 {
        return Err(PyValueError::new_err(
            "procurement_removal_inventory rollout expects input_dim = 7",
        ));
    }
    if config.action_spec.action_dim != 2 {
        return Err(PyValueError::new_err(
            "procurement_removal_inventory rollout expects a two-dimensional action spec",
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

fn policy_state(
    state: &ProcurementRemovalState,
    expected_demand: f64,
    periods: usize,
    returnable_purchase_cap: usize,
) -> PyResult<Vec<f32>> {
    if !expected_demand.is_finite() || expected_demand < 0.0 {
        return Err(PyValueError::new_err(
            "expected_demand must be finite and non-negative",
        ));
    }
    let raw_state = build_raw_state(state)?;
    let inventory_level = raw_state[0];
    let returnable_inventory = raw_state[1];
    let period = raw_state[2] as usize;
    let non_returnable_inventory = inventory_level - returnable_inventory;
    let scale = returnable_purchase_cap
        .max(expected_demand.ceil() as usize)
        .max(1) as f32;
    let remaining_fraction = if periods == 0 {
        0.0
    } else {
        (periods.saturating_sub(period) as f32) / periods as f32
    };
    Ok(vec![
        inventory_level / scale,
        returnable_inventory / scale,
        non_returnable_inventory / scale,
        if inventory_level > 0.0 {
            returnable_inventory / inventory_level
        } else {
            0.0
        },
        expected_demand as f32 / scale,
        returnable_purchase_cap as f32 / scale,
        remaining_fraction,
    ])
}

pub fn rollout(
    flat_params: &[f32],
    config: &ProcurementRemovalRolloutConfig,
    initial_state: &ProcurementRemovalState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;

    for period in 0..config.periods {
        let policy_state = policy_state(
            &state,
            config.demand_mean,
            config.periods,
            config.returnable_purchase_cap,
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
        let (purchase_quantity, removal_quantity) = clip_action(
            &state,
            raw_action[0],
            raw_action[1],
            config.max_purchase_quantity,
            config.max_removal_quantity,
        )?;
        let realized_demand = sample_demand(&mut rng, config.demand_mean, config.demand_kind)?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            removal_quantity,
            realized_demand,
            config.returnable_purchase_cap,
            config.purchase_cost_per_unit,
            config.return_value_per_unit,
            config.liquidation_value_per_unit,
            config.holding_cost_per_unit,
            config.shortage_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= config.discount_factor.powi(config.periods as i32)
        * terminal_salvage_credit(
            &state,
            config.return_value_per_unit,
            config.liquidation_value_per_unit,
        )?;
    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &ProcurementRemovalRolloutConfig,
    initial_state: &ProcurementRemovalState,
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
    config: &ProcurementRemovalRolloutConfig,
    initial_state: &ProcurementRemovalState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(initial_state)?;
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    for (period, demand) in demands.iter().copied().enumerate() {
        let policy_state = policy_state(
            &state,
            config.demand_mean,
            demands.len(),
            config.returnable_purchase_cap,
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
        let (purchase_quantity, removal_quantity) = clip_action(
            &state,
            raw_action[0],
            raw_action[1],
            config.max_purchase_quantity,
            config.max_removal_quantity,
        )?;
        let outcome = step_state(
            &state,
            purchase_quantity,
            removal_quantity,
            demand,
            config.returnable_purchase_cap,
            config.purchase_cost_per_unit,
            config.return_value_per_unit,
            config.liquidation_value_per_unit,
            config.holding_cost_per_unit,
            config.shortage_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }
    discounted_cost -= config.discount_factor.powi(demands.len() as i32)
        * terminal_salvage_credit(
            &state,
            config.return_value_per_unit,
            config.liquidation_value_per_unit,
        )?;
    Ok(discounted_cost)
}
