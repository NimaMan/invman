use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::dense::{
    linear_action_from_flat_params, mlp_action_from_flat_params, ActivationKind, DensePolicyHead,
};
use crate::core::policies::soft_tree::{
    uncapped_scalar_action_from_flat_params, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::lost_sales::demand::{
    build_demand_process, sample_demand, LostSalesDemandConfig,
};
use crate::problems::lost_sales::env::{
    build_pipeline_state, epoch_cost, initialize_state, LostSalesState,
};

#[derive(Clone, Copy)]
pub struct LostSalesRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub demand_config: LostSalesDemandConfig,
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

#[derive(Clone)]
pub struct LostSalesLinearRolloutConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub policy_max_quantity: Option<usize>,
    pub policy_head: DensePolicyHead,
    pub demand_config: LostSalesDemandConfig,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
}

#[derive(Clone)]
pub struct LostSalesNeuralRolloutConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub policy_max_quantity: Option<usize>,
    pub policy_head: DensePolicyHead,
    pub demand_config: LostSalesDemandConfig,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub activation: ActivationKind,
}

fn validate_config(config: &LostSalesRolloutConfig) -> PyResult<()> {
    if config.lead_time < 1 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    if config.input_dim != config.lead_time {
        return Err(PyValueError::new_err(
            "input_dim must match lead_time for pipeline state",
        ));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    if config.leaf_type != SoftTreeLeafType::Linear {
        return Err(PyValueError::new_err(
            "lost-sales soft-tree rollout only supports uncapped linear leaves",
        ));
    }
    Ok(())
}

fn validate_linear_config(config: &LostSalesLinearRolloutConfig) -> PyResult<()> {
    if config.lead_time < 1 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    if config.input_dim != config.lead_time {
        return Err(PyValueError::new_err(
            "input_dim must match lead_time for pipeline state",
        ));
    }
    let expected_output_dim = match config.policy_head {
        DensePolicyHead::CategoricalQuantity => config.output_dim,
        DensePolicyHead::SoftGatedOrdinalQuantity
        | DensePolicyHead::HardGatedOrdinalQuantity => {
            if config.output_dim < 2 {
                return Err(PyValueError::new_err(
                    "output_dim must be at least 2 for the selected policy head",
                ));
            }
            config.output_dim
        }
        DensePolicyHead::DirectQuantity
        | DensePolicyHead::CappedDirectQuantity
        | DensePolicyHead::SigmoidDirectQuantity => 1,
        DensePolicyHead::SoftGatedDirectQuantity
        | DensePolicyHead::GatedSigmoidDirectQuantity
        | DensePolicyHead::HardGatedDirectQuantity => 2,
    };
    if config.output_dim != expected_output_dim {
        return Err(PyValueError::new_err(format!(
            "output_dim must equal {expected_output_dim} for the selected policy head"
        )));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_neural_config(config: &LostSalesNeuralRolloutConfig) -> PyResult<()> {
    if config.hidden_dims.is_empty() {
        return Err(PyValueError::new_err("hidden_dims must be non-empty"));
    }
    validate_linear_config(&LostSalesLinearRolloutConfig {
        input_dim: config.input_dim,
        output_dim: config.output_dim,
        policy_max_quantity: config.policy_max_quantity,
        policy_head: config.policy_head,
        demand_config: config.demand_config,
        lead_time: config.lead_time,
        holding_cost: config.holding_cost,
        shortage_cost: config.shortage_cost,
        procurement_cost: config.procurement_cost,
        fixed_order_cost: config.fixed_order_cost,
        horizon: config.horizon,
        warm_up_periods_ratio: config.warm_up_periods_ratio,
    })
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

fn demand_state_scale(demand_config: &LostSalesDemandConfig) -> PyResult<f64> {
    Ok(demand_config.implied_mean().map_err(PyValueError::new_err)?.max(1.0))
}

pub fn rollout(flat_params: &[f32], config: &LostSalesRolloutConfig, seed: u64) -> PyResult<f64> {
    validate_config(config)?;
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut demand_process =
        build_demand_process(config.demand_config, &mut rng).map_err(PyValueError::new_err)?;
    let mut env_state = initialize_state(
        state_scale,
        config.lead_time,
        &mut rng,
        &mut demand_process,
    );
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = uncapped_scalar_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
        )?;

        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let demand = sample_demand(&mut rng, &mut demand_process);
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
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
        return Err(PyValueError::new_err(
            "input_dim must match lead_time_orders length",
        ));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut epoch_costs = Vec::with_capacity(demands.len());
    for demand in demands.iter() {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = uncapped_scalar_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.depth,
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
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

pub fn linear_rollout(
    flat_params: &[f32],
    config: &LostSalesLinearRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate_linear_config(config)?;
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut demand_process =
        build_demand_process(config.demand_config, &mut rng).map_err(PyValueError::new_err)?;
    let mut env_state = initialize_state(
        state_scale,
        config.lead_time,
        &mut rng,
        &mut demand_process,
    );
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = linear_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.output_dim,
            config.policy_head,
            config.policy_max_quantity,
        )?;
        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;
        let demand = sample_demand(&mut rng, &mut demand_process);
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn linear_rollout_from_demands(
    flat_params: &[f32],
    config: &LostSalesLinearRolloutConfig,
    mut env_state: LostSalesState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_linear_config(config)?;
    if env_state.lead_time_orders.len() != config.input_dim {
        return Err(PyValueError::new_err(
            "lead_time_orders length must match input_dim",
        ));
    }
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut epoch_costs = Vec::with_capacity(demands.len());
    for demand in demands.iter() {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = linear_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            config.output_dim,
            config.policy_head,
            config.policy_max_quantity,
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn linear_population_rollout(
    params_batch: &[Vec<f32>],
    config: &LostSalesLinearRolloutConfig,
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
        .map(|(flat_params, seed)| linear_rollout(flat_params, config, *seed))
        .collect();
    let mut costs = Vec::with_capacity(results.len());
    for result in results {
        costs.push(result?);
    }
    Ok(costs)
}

pub fn neural_rollout(
    flat_params: &[f32],
    config: &LostSalesNeuralRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate_neural_config(config)?;
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut demand_process =
        build_demand_process(config.demand_config, &mut rng).map_err(PyValueError::new_err)?;
    let mut env_state = initialize_state(
        state_scale,
        config.lead_time,
        &mut rng,
        &mut demand_process,
    );
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = mlp_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            &config.hidden_dims,
            config.output_dim,
            config.activation,
            config.policy_head,
            config.policy_max_quantity,
        )?;
        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;
        let demand = sample_demand(&mut rng, &mut demand_process);
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn neural_rollout_from_demands(
    flat_params: &[f32],
    config: &LostSalesNeuralRolloutConfig,
    mut env_state: LostSalesState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_neural_config(config)?;
    if env_state.lead_time_orders.len() != config.input_dim {
        return Err(PyValueError::new_err(
            "lead_time_orders length must match input_dim",
        ));
    }
    let state_scale = demand_state_scale(&config.demand_config)?;

    let mut epoch_costs = Vec::with_capacity(demands.len());
    for demand in demands.iter() {
        let state = build_pipeline_state(
            env_state.current_inventory,
            &env_state.lead_time_orders,
            state_scale,
        );
        let action = mlp_action_from_flat_params(
            &state,
            flat_params,
            config.input_dim,
            &config.hidden_dims,
            config.output_dim,
            config.activation,
            config.policy_head,
            config.policy_max_quantity,
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

    Ok(mean_after_warmup(
        &epoch_costs,
        config.warm_up_periods_ratio,
    ))
}

pub fn neural_population_rollout(
    params_batch: &[Vec<f32>],
    config: &LostSalesNeuralRolloutConfig,
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
        .map(|(flat_params, seed)| neural_rollout(flat_params, config, *seed))
        .collect();
    let mut costs = Vec::with_capacity(results.len());
    for result in results {
        costs.push(result?);
    }
    Ok(costs)
}
