use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::network_inventory::demand::{
    parse_demand_distribution_kind, DemandModel,
};
use crate::problems::network_inventory::env::{NetworkEdge, NetworkInventoryGraph};
use crate::problems::network_inventory::heuristics::{
    node_base_stock_requests, policy_rollout_from_paths,
};
use crate::problems::network_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    NetworkInventoryRolloutConfig,
};

fn build_graph(
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
) -> PyResult<NetworkInventoryGraph> {
    if edge_from.len() != edge_to.len() || edge_from.len() != edge_lead_times.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edge_from, edge_to, and edge_lead_times must have the same length",
        ));
    }
    Ok(NetworkInventoryGraph {
        num_nodes,
        source_nodes,
        edges: edge_from
            .iter()
            .zip(edge_to.iter())
            .zip(edge_lead_times.iter())
            .map(|((from, to), lead_time)| NetworkEdge {
                from: *from,
                to: *to,
                lead_time: *lead_time,
            })
            .collect(),
    })
}

fn build_demand_models(
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
) -> PyResult<Vec<DemandModel>> {
    if demand_kinds.len() != demand_means.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_kinds and demand_means must have the same length",
        ));
    }
    demand_kinds
        .iter()
        .zip(demand_means.iter())
        .map(|(kind, mean)| {
            Ok(DemandModel {
                kind: parse_demand_distribution_kind(kind)?,
                param1: *mean,
            })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    num_nodes,
    source_nodes,
    edge_from,
    edge_to,
    edge_lead_times,
    on_hand_inventory,
    backlog,
    edge_pipelines,
    periods,
    demand_kinds,
    demand_means,
    holding_costs,
    backlog_costs,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn network_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let graph = build_graph(num_nodes, source_nodes, edge_from, edge_to, edge_lead_times)?;
    let initial_state = build_initial_state(&graph, &on_hand_inventory, &backlog, &edge_pipelines)?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        graph,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout(&flat_params, &config, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    num_nodes,
    source_nodes,
    edge_from,
    edge_to,
    edge_lead_times,
    on_hand_inventory,
    backlog,
    edge_pipelines,
    periods,
    demand_kinds,
    demand_means,
    holding_costs,
    backlog_costs,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn network_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let graph = build_graph(num_nodes, source_nodes, edge_from, edge_to, edge_lead_times)?;
    let initial_state = build_initial_state(&graph, &on_hand_inventory, &backlog, &edge_pipelines)?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        graph,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    population_rollout(&params_batch, &config, &initial_state, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    num_nodes,
    source_nodes,
    edge_from,
    edge_to,
    edge_lead_times,
    on_hand_inventory,
    backlog,
    edge_pipelines,
    realized_demands,
    demand_kinds,
    demand_means,
    holding_costs,
    backlog_costs,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn network_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
    realized_demands: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let graph = build_graph(num_nodes, source_nodes, edge_from, edge_to, edge_lead_times)?;
    let initial_state = build_initial_state(&graph, &on_hand_inventory, &backlog, &edge_pipelines)?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_demands.len(),
        graph,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &realized_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    num_nodes,
    source_nodes,
    edge_from,
    edge_to,
    edge_lead_times,
    on_hand_inventory,
    backlog,
    edge_pipelines,
    realized_demands,
    holding_costs,
    backlog_costs,
    discount_factor=0.99
))]
fn network_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
    realized_demands: Vec<Vec<usize>>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let graph = build_graph(num_nodes, source_nodes, edge_from, edge_to, edge_lead_times)?;
    let initial_state = build_initial_state(&graph, &on_hand_inventory, &backlog, &edge_pipelines)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &graph,
        &initial_state,
        &realized_demands,
        &holding_costs,
        &backlog_costs,
        discount_factor,
    )
}

#[pyfunction]
fn network_inventory_node_base_stock_requests(
    num_nodes: usize,
    source_nodes: Vec<bool>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
    base_stock_levels: Vec<usize>,
) -> PyResult<Vec<usize>> {
    let graph = build_graph(num_nodes, source_nodes, edge_from, edge_to, edge_lead_times)?;
    let state = build_initial_state(&graph, &on_hand_inventory, &backlog, &edge_pipelines)?;
    node_base_stock_requests(&graph, &state, &base_stock_levels)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(network_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_node_base_stock_requests,
        m
    )?)?;
    Ok(())
}
