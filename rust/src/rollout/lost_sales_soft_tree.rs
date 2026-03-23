use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

use crate::env::lost_sales::{build_pipeline_state, epoch_cost, initialize_state, LostSalesState};
use crate::policies::soft_tree::{action_from_flat_params, SoftTreeLeafType, SoftTreeSplitType};

#[derive(Clone, Copy)]
pub struct LostSalesRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub max_order_size: usize,
    pub demand_rate: f64,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_config(config: &LostSalesRolloutConfig) -> PyResult<()> {
    if config.lead_time < 1 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    if config.input_dim != config.lead_time {
        return Err(PyValueError::new_err("input_dim must match lead_time for pipeline state"));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err("warm_up_periods_ratio must be in [0, 1]"));
    }
    Ok(())
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

pub fn rollout(
    flat_params: &[f32],
    config: &LostSalesRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let demand_dist = Poisson::new(config.demand_rate)
        .map_err(|err| PyValueError::new_err(format!("invalid demand_rate: {err}")))?;
    let mut env_state =
        initialize_state(config.demand_rate, config.lead_time, config.max_order_size, &mut rng, &demand_dist);
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            config.max_order_size,
        );
        let action = action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
            config.max_order_size,
            config.temperature,
            config.split_type,
            config.leaf_type,
        )?;

        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let demand = demand_dist.sample(&mut rng) as i64;
        let cost = epoch_cost(
            &mut env_state.current_inventory,
            demand,
            action,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
        );
        epoch_costs.push(cost);
    }

    Ok(mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio))
}

pub fn rollout_from_demands(
    flat_params: &[f32],
    config: &LostSalesRolloutConfig,
    mut env_state: LostSalesState,
    demands: &[usize],
) -> PyResult<f64> {
    if env_state.lead_time_orders.is_empty() {
        return Err(PyValueError::new_err("lead_time_orders must be non-empty"));
    }
    if config.input_dim != env_state.lead_time_orders.len() {
        return Err(PyValueError::new_err("input_dim must match lead_time_orders length"));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err("warm_up_periods_ratio must be in [0, 1]"));
    }

    let mut epoch_costs = Vec::with_capacity(demands.len());
    for demand in demands.iter() {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            config.max_order_size,
        );
        let action = action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
            config.max_order_size,
            config.temperature,
            config.split_type,
            config.leaf_type,
        )?;

        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let cost = epoch_cost(
            &mut env_state.current_inventory,
            *demand as i64,
            action,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
        );
        epoch_costs.push(cost);
    }

    Ok(mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio))
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &LostSalesRolloutConfig,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(format!(
            "params batch size {} does not match seeds size {}",
            params_batch.len(),
            seeds.len()
        )));
    }

    let results: Vec<PyResult<f64>> = params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| rollout(flat_params, config, *seed))
        .collect();

    let mut costs = Vec::with_capacity(results.len());
    for result in results {
        costs.push(result?);
    }
    Ok(costs)
}
