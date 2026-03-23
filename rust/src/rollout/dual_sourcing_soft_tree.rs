use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::env::dual_sourcing::{epoch_cost, initialize_state, step_state, validate_action};
use crate::policies::soft_tree::{
    action_vector_from_flat_params, dual_sourcing_action_from_controls, SoftTreeActionAdapter, SoftTreeActionSpec, SoftTreeLeafType,
    SoftTreeSplitType,
};

#[derive(Clone)]
pub struct DualSourcingRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub regular_lead_time: usize,
    pub regular_order_cost: f64,
    pub expedited_order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub regular_max_order_size: usize,
    pub expedited_max_order_size: usize,
    pub demand_low: usize,
    pub demand_high: usize,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
    pub action_adapter: SoftTreeActionAdapter,
}

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

pub fn rollout(flat_params: &[f32], config: &DualSourcingRolloutConfig, seed: u64) -> PyResult<f64> {
    if config.input_dim != config.regular_lead_time {
        return Err(PyValueError::new_err("dual-sourcing rollout expects input_dim == regular_lead_time"));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env_state = initialize_state(
        config.regular_lead_time,
        config.regular_max_order_size,
        config.demand_low,
        config.demand_high,
        seed,
    );
    let mut epoch_costs = Vec::with_capacity(config.horizon);
    let scale = (config.regular_max_order_size + config.expedited_max_order_size).max(1) as f32;

    for _ in 0..config.horizon {
        let state = env_state
            .reduced_state
            .iter()
            .map(|value| *value as f32 / scale)
            .collect::<Vec<_>>();
        let controls = action_vector_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?;
        let action = dual_sourcing_action_from_controls(
            &env_state.reduced_state,
            &controls,
            config.action_adapter,
            config.regular_max_order_size,
            config.expedited_max_order_size,
        )?;
        let regular_order = action[0];
        let expedited_order = action[1];
        validate_action(
            regular_order,
            expedited_order,
            config.regular_max_order_size,
            config.expedited_max_order_size,
        )?;
        let demand = rng.gen_range(config.demand_low..=config.demand_high);
        epoch_costs.push(epoch_cost(
            &env_state.reduced_state,
            regular_order,
            expedited_order,
            demand,
            config.regular_order_cost,
            config.expedited_order_cost,
            config.holding_cost,
            config.shortage_cost,
        ));
        env_state.reduced_state = step_state(&env_state.reduced_state, regular_order, expedited_order, demand);
    }

    Ok(mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio))
}

pub fn rollout_from_demands(
    flat_params: &[f32],
    config: &DualSourcingRolloutConfig,
    mut reduced_state: Vec<i64>,
    demands: &[usize],
) -> PyResult<f64> {
    let mut epoch_costs = Vec::with_capacity(demands.len());
    let scale = (config.regular_max_order_size + config.expedited_max_order_size).max(1) as f32;
    for demand in demands.iter().copied() {
        let state = reduced_state
            .iter()
            .map(|value| *value as f32 / scale)
            .collect::<Vec<_>>();
        let controls = action_vector_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?;
        let action = dual_sourcing_action_from_controls(
            &reduced_state,
            &controls,
            config.action_adapter,
            config.regular_max_order_size,
            config.expedited_max_order_size,
        )?;
        let regular_order = action[0];
        let expedited_order = action[1];
        validate_action(
            regular_order,
            expedited_order,
            config.regular_max_order_size,
            config.expedited_max_order_size,
        )?;
        epoch_costs.push(epoch_cost(
            &reduced_state,
            regular_order,
            expedited_order,
            demand,
            config.regular_order_cost,
            config.expedited_order_cost,
            config.holding_cost,
            config.shortage_cost,
        ));
        reduced_state = step_state(&reduced_state, regular_order, expedited_order, demand);
    }
    Ok(mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio))
}

pub fn population_rollout(params_batch: &[Vec<f32>], config: &DualSourcingRolloutConfig, seeds: &[u64]) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err("params batch size must match seeds size"));
    }
    let results: Vec<PyResult<f64>> = params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(params, seed)| rollout(params, config, *seed))
        .collect();

    let mut costs = Vec::with_capacity(results.len());
    for result in results {
        costs.push(result?);
    }
    Ok(costs)
}
