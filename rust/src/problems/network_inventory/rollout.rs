use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::network_inventory::demand::{
    sample_demand, validate_demand_model, DemandModel,
};
use crate::problems::network_inventory::env::{
    build_policy_state, initialize_state, step_state, supply_relation_count, validate_state,
    NetworkInventoryGraph, NetworkInventoryState,
};

#[derive(Clone)]
pub struct NetworkInventoryRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub graph: NetworkInventoryGraph,
    pub demand_models: Vec<DemandModel>,
    pub holding_costs: Vec<f64>,
    pub backlog_costs: Vec<f64>,
    pub discount_factor: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

pub fn build_initial_state(
    graph: &NetworkInventoryGraph,
    finished_inventory: &[usize],
    raw_inventory_by_relation: &[usize],
    internal_backlog_by_edge: &[usize],
    external_backlog: &[usize],
    supply_pipelines: &[Vec<usize>],
) -> PyResult<NetworkInventoryState> {
    initialize_state(
        graph,
        finished_inventory,
        raw_inventory_by_relation,
        internal_backlog_by_edge,
        external_backlog,
        supply_pipelines,
    )
}

fn validate_config(
    config: &NetworkInventoryRolloutConfig,
    initial_state: &NetworkInventoryState,
) -> PyResult<()> {
    validate_state(&config.graph, initial_state)?;
    if config.demand_models.len() != config.graph.num_nodes
        || config.holding_costs.len() != config.graph.num_nodes
        || config.backlog_costs.len() != config.graph.num_nodes
    {
        return Err(PyValueError::new_err(
            "all node-wise config vectors must match num_nodes",
        ));
    }
    for model in config.demand_models.iter() {
        validate_demand_model(model)?;
    }
    let zero_demands = vec![0usize; config.graph.num_nodes];
    let demand_means = config
        .demand_models
        .iter()
        .map(|model| model.param1)
        .collect::<Vec<_>>();
    let expected_input_dim = build_policy_state(
        &config.graph,
        initial_state,
        &demand_means,
        &zero_demands,
        config.periods,
    )?
    .len();
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected {}",
            config.input_dim, expected_input_dim
        )));
    }
    if config.action_spec.action_dim != supply_relation_count(&config.graph) {
        return Err(PyValueError::new_err(format!(
            "action_spec.action_dim {} does not match the number of supply relations {}",
            config.action_spec.action_dim,
            supply_relation_count(&config.graph)
        )));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

fn edge_requests(
    flat_params: &[f32],
    state: &NetworkInventoryState,
    realized_external_demands: &[usize],
    config: &NetworkInventoryRolloutConfig,
) -> PyResult<Vec<usize>> {
    let demand_means = config
        .demand_models
        .iter()
        .map(|model| model.param1)
        .collect::<Vec<_>>();
    let policy_state = build_policy_state(
        &config.graph,
        state,
        &demand_means,
        realized_external_demands,
        config.periods,
    )?;
    action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )
}

pub fn rollout(
    flat_params: &[f32],
    config: &NetworkInventoryRolloutConfig,
    initial_state: &NetworkInventoryState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let realized_demands = config
            .demand_models
            .iter()
            .map(|model| sample_demand(&mut rng, model))
            .collect::<PyResult<Vec<_>>>()?;
        let requests = edge_requests(flat_params, &state, &realized_demands, config)?;
        let outcome = step_state(
            &config.graph,
            &state,
            &requests,
            &realized_demands,
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
    config: &NetworkInventoryRolloutConfig,
    initial_state: &NetworkInventoryState,
    realized_demands: &[Vec<usize>],
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if realized_demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "realized_demands length must match config.periods",
        ));
    }
    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in realized_demands.iter() {
        if demand.len() != config.graph.num_nodes {
            return Err(PyValueError::new_err(
                "each realized demand vector must match num_nodes",
            ));
        }
        let requests = edge_requests(flat_params, &state, demand, config)?;
        let outcome = step_state(
            &config.graph,
            &state,
            &requests,
            demand,
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
    config: &NetworkInventoryRolloutConfig,
    initial_state: &NetworkInventoryState,
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
