use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::perishable_inventory::env::{
    build_raw_state, initialize_state, step_state, validate_state, IssuingPolicy, PerishableState,
};
use crate::problems::perishable_inventory::heuristics::PolicyTraceSummary;

#[derive(Clone)]
pub struct PerishableInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub demand_mean: f64,
    pub demand_cov: f64,
    pub shelf_life: usize,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub waste_cost: f64,
    pub procurement_cost: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
    pub issuing_policy: IssuingPolicy,
}

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    Ok(active_costs.iter().sum::<f64>() / active_costs.len() as f64)
}

fn discounted_return_after_warmup(
    epoch_costs: &[f64],
    warm_up_periods_ratio: f64,
    gamma: f64,
) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(PyValueError::new_err("gamma must be in [0, 1]"));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let mut discounted_return = 0.0;
    for (offset, cost) in epoch_costs.iter().skip(warm_up_periods).enumerate() {
        discounted_return += -cost * gamma.powi(offset as i32);
    }
    Ok(discounted_return)
}

fn validate_config(config: &PerishableInventoryRolloutConfig) -> PyResult<()> {
    if config.shelf_life < 1 {
        return Err(PyValueError::new_err("shelf_life must be at least 1"));
    }
    if config.lead_time < 1 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    let expected_input_dim = config.shelf_life + config.lead_time.saturating_sub(1);
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim must equal shelf_life + lead_time - 1 = {expected_input_dim}"
        )));
    }
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "perishable-inventory rollout expects a scalar action spec",
        ));
    }
    if config.demand_mean < 0.0 {
        return Err(PyValueError::new_err("demand_mean must be non-negative"));
    }
    if config.demand_cov <= 0.0 {
        return Err(PyValueError::new_err("demand_cov must be positive"));
    }
    if config.horizon < 1 {
        return Err(PyValueError::new_err("horizon must be positive"));
    }
    Ok(())
}

fn build_discrete_gamma(demand_mean: f64, demand_cov: f64) -> PyResult<Gamma<f64>> {
    let shape = 1.0 / (demand_cov * demand_cov);
    let scale = if shape > 0.0 {
        demand_mean / shape
    } else {
        0.0
    };
    Gamma::new(shape, scale.max(1e-9))
        .map_err(|err| PyValueError::new_err(format!("invalid gamma demand parameters: {err}")))
}

fn sample_gamma_demand(rng: &mut StdRng, gamma: &Gamma<f64>) -> usize {
    gamma.sample(rng).round().max(0.0) as usize
}

fn policy_state(state: &PerishableState, demand_mean: f64) -> Vec<f32> {
    let scale = demand_mean.max(1.0) as f32;
    build_raw_state(state)
        .into_iter()
        .map(|value| value / scale)
        .collect()
}

pub fn rollout(
    flat_params: &[f32],
    config: &PerishableInventoryRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config)?;
    let gamma = build_discrete_gamma(config.demand_mean, config.demand_cov)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initialize_state(config.demand_mean, config.shelf_life, config.lead_time);
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let policy_state = policy_state(&state, config.demand_mean);
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?[0];
        let demand = sample_gamma_demand(&mut rng, &gamma);
        let outcome = step_state(
            &state,
            action,
            demand,
            config.holding_cost,
            config.shortage_cost,
            config.waste_cost,
            config.procurement_cost,
            config.issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio)
}

pub fn rollout_discounted_return(
    flat_params: &[f32],
    config: &PerishableInventoryRolloutConfig,
    seed: u64,
    gamma: f64,
) -> PyResult<f64> {
    validate_config(config)?;
    let gamma_dist = build_discrete_gamma(config.demand_mean, config.demand_cov)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initialize_state(config.demand_mean, config.shelf_life, config.lead_time);
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let policy_state = policy_state(&state, config.demand_mean);
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?[0];
        let demand = sample_gamma_demand(&mut rng, &gamma_dist);
        let outcome = step_state(
            &state,
            action,
            demand,
            config.holding_cost,
            config.shortage_cost,
            config.waste_cost,
            config.procurement_cost,
            config.issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    discounted_return_after_warmup(&epoch_costs, config.warm_up_periods_ratio, gamma)
}

pub fn rollout_from_demands(
    flat_params: &[f32],
    config: &PerishableInventoryRolloutConfig,
    mut state: PerishableState,
    demands: &[usize],
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(&state, config.shelf_life, config.lead_time)?;
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter().copied() {
        let policy_state = policy_state(&state, config.demand_mean);
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?[0];
        let outcome = step_state(
            &state,
            action,
            demand,
            config.holding_cost,
            config.shortage_cost,
            config.waste_cost,
            config.procurement_cost,
            config.issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio)
}

pub fn rollout_trace_summary_from_demands(
    flat_params: &[f32],
    config: &PerishableInventoryRolloutConfig,
    mut state: PerishableState,
    demands: &[usize],
) -> PyResult<PolicyTraceSummary> {
    validate_config(config)?;
    validate_state(&state, config.shelf_life, config.lead_time)?;
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }

    let mut total_cost = 0.0;
    let mut total_demand = 0usize;
    let mut total_shortage = 0usize;
    let mut stockout_periods = 0usize;
    let mut total_waste = 0usize;
    let mut total_holding_inventory = 0usize;
    let mut total_order_quantity = 0usize;
    let mut positive_order_periods = 0usize;

    for demand in demands.iter().copied() {
        let policy_state = policy_state(&state, config.demand_mean);
        let order_quantity = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?[0];
        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            config.holding_cost,
            config.shortage_cost,
            config.waste_cost,
            config.procurement_cost,
            config.issuing_policy,
        );
        total_cost += outcome.cost;
        total_demand += demand;
        total_shortage += outcome.shortage;
        total_waste += outcome.waste;
        total_holding_inventory += outcome.holding_inventory;
        total_order_quantity += order_quantity;
        if order_quantity > 0 {
            positive_order_periods += 1;
        }
        if outcome.shortage > 0 {
            stockout_periods += 1;
        }
        state = outcome.next_state;
    }

    let periods = demands.len();
    let ending_inventory = state.on_hand.iter().copied().sum::<usize>();
    let ending_pipeline = state.pipeline_orders.iter().copied().sum::<usize>();

    Ok(PolicyTraceSummary {
        periods,
        total_cost,
        mean_period_cost: total_cost / periods as f64,
        total_demand,
        total_shortage,
        fill_rate: if total_demand > 0 {
            1.0 - total_shortage as f64 / total_demand as f64
        } else {
            1.0
        },
        cycle_service_level: 1.0 - stockout_periods as f64 / periods as f64,
        total_waste,
        waste_rate: if total_demand > 0 {
            total_waste as f64 / total_demand as f64
        } else {
            0.0
        },
        mean_holding_inventory: total_holding_inventory as f64 / periods as f64,
        mean_order_quantity: total_order_quantity as f64 / periods as f64,
        positive_order_frequency: positive_order_periods as f64 / periods as f64,
        ending_inventory,
        ending_pipeline,
    })
}

pub fn rollout_from_demands_discounted_return(
    flat_params: &[f32],
    config: &PerishableInventoryRolloutConfig,
    mut state: PerishableState,
    demands: &[usize],
    gamma: f64,
) -> PyResult<f64> {
    validate_config(config)?;
    validate_state(&state, config.shelf_life, config.lead_time)?;
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter().copied() {
        let policy_state = policy_state(&state, config.demand_mean);
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?[0];
        let outcome = step_state(
            &state,
            action,
            demand,
            config.holding_cost,
            config.shortage_cost,
            config.waste_cost,
            config.procurement_cost,
            config.issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    discounted_return_after_warmup(&epoch_costs, config.warm_up_periods_ratio, gamma)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &PerishableInventoryRolloutConfig,
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
        .map(|(flat_params, seed)| rollout(flat_params, config, *seed))
        .collect()
}

pub fn population_rollout_discounted_return(
    params_batch: &[Vec<f32>],
    config: &PerishableInventoryRolloutConfig,
    seeds: &[u64],
    gamma: f64,
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| rollout_discounted_return(flat_params, config, *seed, gamma))
        .collect()
}
