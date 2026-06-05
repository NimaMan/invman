use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::decentralized_inventory_control::demand::{
    sample_demand, validate_demand_model, DemandModel,
};
use crate::problems::decentralized_inventory_control::env::{
    build_local_policy_state, initialize_state, step_state, validate_state,
    DecentralizedInventoryControlState,
};

#[derive(Clone)]
pub struct DecentralizedInventoryControlRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub customer_demand_model: DemandModel,
    pub demand_smoothing_factors: Vec<f64>,
    pub holding_costs: Vec<f64>,
    pub backlog_costs: Vec<f64>,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

pub fn build_initial_state(
    on_hand_inventory: &[usize],
    backlog: &[usize],
    shipment_pipelines: &[Vec<usize>],
    order_pipelines: &[Vec<usize>],
    last_received_shipments: &[usize],
    last_received_orders: &[usize],
    forecast_orders: &[f64],
    last_actions: &[usize],
) -> PyResult<DecentralizedInventoryControlState> {
    initialize_state(
        on_hand_inventory,
        backlog,
        shipment_pipelines,
        order_pipelines,
        last_received_shipments,
        last_received_orders,
        forecast_orders,
        last_actions,
    )
}

fn validate_config(
    config: &DecentralizedInventoryControlRolloutConfig,
    initial_state: &DecentralizedInventoryControlState,
) -> PyResult<()> {
    validate_state(initial_state)?;
    validate_demand_model(&config.customer_demand_model)?;
    let num_agents = initial_state.on_hand_inventory.len();
    if config.action_spec.action_dim != 1 {
        return Err(PyValueError::new_err(
            "decentralized_inventory_control rollout expects a one-dimensional action spec",
        ));
    }
    if config.input_dim != 12 {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected local policy state size 12",
            config.input_dim
        )));
    }
    if config.demand_smoothing_factors.len() != num_agents
        || config.holding_costs.len() != num_agents
        || config.backlog_costs.len() != num_agents
    {
        return Err(PyValueError::new_err(
            "all per-agent config vectors must match the number of agents",
        ));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

fn action_vector(
    flat_params: &[f32],
    state: &DecentralizedInventoryControlState,
    realized_customer_demand: usize,
    config: &DecentralizedInventoryControlRolloutConfig,
) -> PyResult<Vec<usize>> {
    (0..state.on_hand_inventory.len())
        .map(|agent_idx| {
            let local_state = build_local_policy_state(
                state,
                agent_idx,
                config.periods,
                &config.holding_costs,
                &config.backlog_costs,
                realized_customer_demand,
            )?;
            let action = action_vector_from_flat_params(
                &local_state,
                flat_params,
                config.input_dim,
                config.depth,
                config.temperature,
                config.split_type,
                config.leaf_type,
                &config.action_spec,
            )?;
            Ok(action[0])
        })
        .collect()
}

pub fn rollout(
    flat_params: &[f32],
    config: &DecentralizedInventoryControlRolloutConfig,
    initial_state: &DecentralizedInventoryControlState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let customer_demand = sample_demand(&mut rng, &config.customer_demand_model)?;
        let actions = action_vector(flat_params, &state, customer_demand, config)?;
        let outcome = step_state(
            &state,
            &actions,
            customer_demand,
            &config.demand_smoothing_factors,
            &config.holding_costs,
            &config.backlog_costs,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &DecentralizedInventoryControlRolloutConfig,
    initial_state: &DecentralizedInventoryControlState,
    customer_demands: &[usize],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if customer_demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "customer_demands length must match config.periods",
        ));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in customer_demands.iter().copied() {
        let actions = action_vector(flat_params, &state, demand, config)?;
        let outcome = step_state(
            &state,
            &actions,
            demand,
            &config.demand_smoothing_factors,
            &config.holding_costs,
            &config.backlog_costs,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &DecentralizedInventoryControlRolloutConfig,
    initial_state: &DecentralizedInventoryControlState,
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
