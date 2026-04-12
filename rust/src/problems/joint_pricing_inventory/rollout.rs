use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::joint_pricing_inventory::demand::{
    sample_demand, validate_price_ladder, DemandDistributionKind,
};
use crate::problems::joint_pricing_inventory::env::{
    build_raw_state, clip_action, initialize_state, step_state, terminal_salvage_credit,
    JointPricingInventoryState,
};

#[derive(Clone)]
pub struct JointPricingInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_kind: DemandDistributionKind,
    pub price_levels: Vec<f64>,
    pub demand_means: Vec<f64>,
    pub procurement_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub max_order_quantity: usize,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(config: &JointPricingInventoryRolloutConfig) -> PyResult<()> {
    if config.input_dim != 7 {
        return Err(PyValueError::new_err(
            "joint_pricing_inventory rollout expects input_dim = 7",
        ));
    }
    if config.action_spec.action_dim != 2 {
        return Err(PyValueError::new_err(
            "joint_pricing_inventory rollout expects a two-dimensional action spec",
        ));
    }
    if config.periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    validate_price_ladder(&config.price_levels, &config.demand_means)
}

pub fn build_initial_state(inventory_level: usize) -> PyResult<JointPricingInventoryState> {
    initialize_state(inventory_level)
}

fn policy_state(
    state: &JointPricingInventoryState,
    price_levels: &[f64],
    demand_means: &[f64],
    periods: usize,
    max_order_quantity: usize,
) -> PyResult<Vec<f32>> {
    validate_price_ladder(price_levels, demand_means)?;
    let raw_state = build_raw_state(state);
    let inventory_level = raw_state[0];
    let period = raw_state[1] as usize;
    let max_demand_mean = demand_means
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value));
    let mean_demand_mean = demand_means.iter().sum::<f64>() / demand_means.len() as f64;
    let scale = (inventory_level as usize)
        .max(max_order_quantity)
        .max(max_demand_mean.ceil() as usize)
        .max(1) as f32;
    let price_scale = price_levels
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value))
        .max(1.0) as f32;
    let remaining_fraction = if periods == 0 {
        0.0
    } else {
        (periods.saturating_sub(period) as f32) / periods as f32
    };
    Ok(vec![
        inventory_level / scale,
        demand_means[0] as f32 / scale,
        demand_means[demand_means.len() - 1] as f32 / scale,
        mean_demand_mean as f32 / scale,
        price_levels[0] as f32 / price_scale,
        price_levels[price_levels.len() - 1] as f32 / price_scale,
        remaining_fraction,
    ])
}

pub fn rollout(
    flat_params: &[f32],
    config: &JointPricingInventoryRolloutConfig,
    initial_state: &JointPricingInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;

    for period in 0..config.periods {
        let policy_state = policy_state(
            &state,
            &config.price_levels,
            &config.demand_means,
            config.periods,
            config.max_order_quantity,
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
        let (order_quantity, price_index) = clip_action(
            raw_action[0],
            raw_action[1],
            config.max_order_quantity,
            config.price_levels.len(),
        )?;
        let realized_demand = sample_demand(
            &mut rng,
            price_index,
            &config.demand_means,
            config.demand_kind,
        )?;
        let outcome = step_state(
            &state,
            order_quantity,
            price_index,
            realized_demand,
            &config.price_levels,
            config.procurement_cost_per_unit,
            config.holding_cost_per_unit,
            config.stockout_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= config.discount_factor.powi(config.periods as i32)
        * terminal_salvage_credit(&state, config.salvage_value_per_unit)?;
    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &JointPricingInventoryRolloutConfig,
    initial_state: &JointPricingInventoryState,
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
    config: &JointPricingInventoryRolloutConfig,
    initial_state: &JointPricingInventoryState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_config(config)?;
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    for (period, demand) in demands.iter().copied().enumerate() {
        let policy_state = policy_state(
            &state,
            &config.price_levels,
            &config.demand_means,
            demands.len(),
            config.max_order_quantity,
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
        let (order_quantity, price_index) = clip_action(
            raw_action[0],
            raw_action[1],
            config.max_order_quantity,
            config.price_levels.len(),
        )?;
        let outcome = step_state(
            &state,
            order_quantity,
            price_index,
            demand,
            &config.price_levels,
            config.procurement_cost_per_unit,
            config.holding_cost_per_unit,
            config.stockout_cost_per_unit,
        )?;
        discounted_cost += config.discount_factor.powi(period as i32) * outcome.period_cost;
        state = outcome.next_state;
    }

    discounted_cost -= config.discount_factor.powi(demands.len() as i32)
        * terminal_salvage_credit(&state, config.salvage_value_per_unit)?;
    Ok(discounted_cost)
}
