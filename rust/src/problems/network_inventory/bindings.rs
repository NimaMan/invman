use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::core::flownet::{
    PolicyPerformanceVerificationSummary, PolicyScoreOrdering, PolicyVerificationRole,
};
use crate::problems::network_inventory::demand::{parse_demand_distribution_kind, DemandModel};
use crate::problems::network_inventory::env::{
    NetworkEdge, NetworkInventoryGraph, NetworkNodeMode,
};
use crate::problems::network_inventory::flownet::verification::verify_exact_reference_policy_performance;
use crate::problems::network_inventory::heuristics::{
    pairwise_base_stock_requests, policy_rollout_from_paths,
};
use crate::problems::network_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    NetworkInventoryRolloutConfig,
};
use crate::problems::network_inventory::verification::literature_benchmarks::literature_benchmark_summary;

fn build_graph(
    num_nodes: usize,
    source_nodes: Vec<bool>,
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
) -> PyResult<NetworkInventoryGraph> {
    if edge_from.len() != edge_to.len() || edge_from.len() != edge_lead_times.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edge_from, edge_to, and edge_lead_times must have the same length",
        ));
    }
    if node_modes.len() != num_nodes || external_supplier_lead_times.len() != num_nodes {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "node_modes and external_supplier_lead_times must match num_nodes",
        ));
    }
    Ok(NetworkInventoryGraph {
        num_nodes,
        source_nodes,
        node_modes: node_modes
            .iter()
            .map(|mode| match mode.as_str() {
                "single" => Ok(NetworkNodeMode::Single),
                "assembly_and" => Ok(NetworkNodeMode::AssemblyAnd),
                "assembly_or" => Ok(NetworkNodeMode::AssemblyOr),
                _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported node mode '{mode}'"
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?,
        external_supplier_lead_times,
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
    demand_param1s: Vec<f64>,
    demand_param2s: Option<Vec<f64>>,
) -> PyResult<Vec<DemandModel>> {
    if demand_kinds.len() != demand_param1s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_kinds and demand_param1s must have the same length",
        ));
    }
    let param2s = demand_param2s.unwrap_or_else(|| vec![0.0; demand_kinds.len()]);
    if param2s.len() != demand_kinds.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_param2s must match demand_kinds when provided",
        ));
    }
    demand_kinds
        .iter()
        .zip(demand_param1s.iter())
        .zip(param2s.iter())
        .map(|((kind, param1), param2)| {
            Ok(DemandModel {
                kind: parse_demand_distribution_kind(kind)?,
                param1: *param1,
                param2: *param2,
            })
        })
        .collect()
}

fn policy_verification_role_to_str(role: PolicyVerificationRole) -> &'static str {
    match role {
        PolicyVerificationRole::OptimalReference => "optimal_reference",
        PolicyVerificationRole::Heuristic => "heuristic",
        PolicyVerificationRole::LearnedPolicyThreshold => "learned_policy_threshold",
    }
}

fn policy_score_ordering_to_str(ordering: PolicyScoreOrdering) -> &'static str {
    match ordering {
        PolicyScoreOrdering::LowerIsBetter => "lower_is_better",
        PolicyScoreOrdering::HigherIsBetter => "higher_is_better",
    }
}

fn policy_performance_summary_to_py(
    py: Python<'_>,
    summary: &PolicyPerformanceVerificationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", &summary.reference_name)?;
    dict.set_item("horizon_periods", summary.horizon_periods)?;
    dict.set_item(
        "score_ordering",
        policy_score_ordering_to_str(summary.score_ordering),
    )?;
    dict.set_item(
        "all_observed_targets_within_tolerance",
        summary.all_observed_targets_within_tolerance(),
    )?;
    dict.set_item(
        "observed_targets_are_sorted_from_best_to_worst",
        summary.observed_targets_are_sorted_from_best_to_worst(),
    )?;

    let results = summary
        .results
        .iter()
        .map(|result| {
            let result_dict = PyDict::new_bound(py);
            result_dict.set_item("policy_name", &result.target.policy_name)?;
            result_dict.set_item("role", policy_verification_role_to_str(result.target.role))?;
            result_dict.set_item("expected_score", result.target.expected_score)?;
            result_dict.set_item("tolerance", result.target.tolerance)?;
            result_dict.set_item("observed_score", result.observed_score)?;
            result_dict.set_item("abs_gap", result.abs_gap)?;
            result_dict.set_item("within_tolerance", result.within_tolerance)?;
            Ok(result_dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    dict.set_item("results", results)?;

    let untargeted = summary
        .untargeted_measurements
        .iter()
        .map(|measurement| {
            let measurement_dict = PyDict::new_bound(py);
            measurement_dict.set_item("policy_name", &measurement.policy_name)?;
            measurement_dict.set_item("observed_score", measurement.observed_score)?;
            Ok(measurement_dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    dict.set_item("untargeted_measurements", untargeted)?;

    Ok(dict.into_any().unbind().into())
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
    node_modes,
    external_supplier_lead_times,
    edge_from,
    edge_to,
    edge_lead_times,
    finished_inventory,
    raw_inventory_by_relation,
    internal_backlog_by_edge,
    external_backlog,
    supply_pipelines,
    periods,
    demand_kinds,
    demand_param1s,
    holding_costs,
    backlog_costs,
    demand_param2s=None,
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
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_param1s: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    demand_param2s: Option<Vec<f64>>,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let graph = build_graph(
        num_nodes,
        source_nodes,
        node_modes,
        external_supplier_lead_times,
        edge_from,
        edge_to,
        edge_lead_times,
    )?;
    let initial_state = build_initial_state(
        &graph,
        &finished_inventory,
        &raw_inventory_by_relation,
        &internal_backlog_by_edge,
        &external_backlog,
        &supply_pipelines,
    )?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        graph,
        demand_models: build_demand_models(demand_kinds, demand_param1s, demand_param2s)?,
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
    node_modes,
    external_supplier_lead_times,
    edge_from,
    edge_to,
    edge_lead_times,
    finished_inventory,
    raw_inventory_by_relation,
    internal_backlog_by_edge,
    external_backlog,
    supply_pipelines,
    periods,
    demand_kinds,
    demand_param1s,
    holding_costs,
    backlog_costs,
    seeds,
    demand_param2s=None,
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
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_param1s: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    seeds: Vec<u64>,
    demand_param2s: Option<Vec<f64>>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let graph = build_graph(
        num_nodes,
        source_nodes,
        node_modes,
        external_supplier_lead_times,
        edge_from,
        edge_to,
        edge_lead_times,
    )?;
    let initial_state = build_initial_state(
        &graph,
        &finished_inventory,
        &raw_inventory_by_relation,
        &internal_backlog_by_edge,
        &external_backlog,
        &supply_pipelines,
    )?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        graph,
        demand_models: build_demand_models(demand_kinds, demand_param1s, demand_param2s)?,
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
    node_modes,
    external_supplier_lead_times,
    edge_from,
    edge_to,
    edge_lead_times,
    finished_inventory,
    raw_inventory_by_relation,
    internal_backlog_by_edge,
    external_backlog,
    supply_pipelines,
    realized_demands,
    demand_kinds,
    demand_param1s,
    holding_costs,
    backlog_costs,
    demand_param2s=None,
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
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
    realized_demands: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_param1s: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    demand_param2s: Option<Vec<f64>>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let graph = build_graph(
        num_nodes,
        source_nodes,
        node_modes,
        external_supplier_lead_times,
        edge_from,
        edge_to,
        edge_lead_times,
    )?;
    let initial_state = build_initial_state(
        &graph,
        &finished_inventory,
        &raw_inventory_by_relation,
        &internal_backlog_by_edge,
        &external_backlog,
        &supply_pipelines,
    )?;
    let config = NetworkInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_demands.len(),
        graph,
        demand_models: build_demand_models(demand_kinds, demand_param1s, demand_param2s)?,
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
    node_modes,
    external_supplier_lead_times,
    edge_from,
    edge_to,
    edge_lead_times,
    finished_inventory,
    raw_inventory_by_relation,
    internal_backlog_by_edge,
    external_backlog,
    supply_pipelines,
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
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
    realized_demands: Vec<Vec<usize>>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let graph = build_graph(
        num_nodes,
        source_nodes,
        node_modes,
        external_supplier_lead_times,
        edge_from,
        edge_to,
        edge_lead_times,
    )?;
    let initial_state = build_initial_state(
        &graph,
        &finished_inventory,
        &raw_inventory_by_relation,
        &internal_backlog_by_edge,
        &external_backlog,
        &supply_pipelines,
    )?;
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
fn network_inventory_pairwise_base_stock_requests(
    num_nodes: usize,
    source_nodes: Vec<bool>,
    node_modes: Vec<String>,
    external_supplier_lead_times: Vec<usize>,
    edge_from: Vec<usize>,
    edge_to: Vec<usize>,
    edge_lead_times: Vec<usize>,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
    realized_external_demands: Vec<usize>,
    base_stock_levels: Vec<usize>,
) -> PyResult<Vec<usize>> {
    let graph = build_graph(
        num_nodes,
        source_nodes,
        node_modes,
        external_supplier_lead_times,
        edge_from,
        edge_to,
        edge_lead_times,
    )?;
    let state = build_initial_state(
        &graph,
        &finished_inventory,
        &raw_inventory_by_relation,
        &internal_backlog_by_edge,
        &external_backlog,
        &supply_pipelines,
    )?;
    pairwise_base_stock_requests(&graph, &state, &base_stock_levels, &realized_external_demands)
}

#[pyfunction]
fn network_inventory_flownet_policy_verification_summary(py: Python<'_>) -> PyResult<PyObject> {
    let summary = verify_exact_reference_policy_performance()
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    policy_performance_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (serial_replications=10000, seed=1234))]
fn network_inventory_literature_benchmark_summary(
    py: Python<'_>,
    serial_replications: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let summary = literature_benchmark_summary(serial_replications, seed);
    let dict = PyDict::new_bound(py);
    dict.set_item("source", summary.source)?;
    dict.set_item("url", summary.url)?;

    let single_node_results = summary
        .single_node_results
        .iter()
        .map(|row| {
            let row_dict = PyDict::new_bound(py);
            row_dict.set_item("case_idx", row.case_idx)?;
            row_dict.set_item("published_analytical_oul", row.published_analytical_oul)?;
            row_dict.set_item("reproduced_analytical_oul", row.reproduced_analytical_oul)?;
            row_dict.set_item(
                "published_analytical_average_cost",
                row.published_analytical_average_cost,
            )?;
            row_dict.set_item(
                "reproduced_analytical_average_cost",
                row.reproduced_analytical_average_cost,
            )?;
            row_dict.set_item(
                "oul_abs_gap",
                (row.reproduced_analytical_oul - row.published_analytical_oul).abs(),
            )?;
            row_dict.set_item(
                "cost_abs_gap",
                (row.reproduced_analytical_average_cost - row.published_analytical_average_cost)
                    .abs(),
            )?;
            Ok(row_dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    dict.set_item("single_node_results", single_node_results)?;
    dict.set_item(
        "notes",
        "Only the paper's single-node analytical rows are currently reproduced tightly enough for verification. The serial rows remain literature catalog entries only and are not part of the verification summary.",
    )?;

    Ok(dict.into())
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
        network_inventory_pairwise_base_stock_requests,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_flownet_policy_verification_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        network_inventory_literature_benchmark_summary,
        m
    )?)?;
    Ok(())
}
