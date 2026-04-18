use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::decentralized_inventory_control::demand::{
    parse_demand_distribution_kind, DemandModel,
};
use crate::problems::decentralized_inventory_control::heuristics::{
    base_stock_orders, policy_rollout_from_paths, sterman_anchor_adjust_orders,
};
use crate::problems::decentralized_inventory_control::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    DecentralizedInventoryControlRolloutConfig,
};
use crate::problems::decentralized_inventory_control::verification::classic_board_game::simulate_classic_sterman_benchmark;

fn build_customer_demand_model(
    demand_distribution: &str,
    demand_mean: f64,
) -> PyResult<DemandModel> {
    Ok(DemandModel {
        kind: parse_demand_distribution_kind(demand_distribution)?,
        param1: demand_mean,
    })
}

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    periods: usize,
    demand_mean: f64,
    demand_distribution: &str,
    demand_smoothing_factors: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<DecentralizedInventoryControlRolloutConfig> {
    Ok(DecentralizedInventoryControlRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        customer_demand_model: build_customer_demand_model(demand_distribution, demand_mean)?,
        demand_smoothing_factors,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    })
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    on_hand_inventory,
    backlog,
    shipment_pipelines,
    order_pipelines,
    last_received_shipments,
    last_received_orders,
    forecast_orders,
    last_actions,
    periods,
    demand_mean,
    demand_smoothing_factors,
    holding_costs,
    backlog_costs,
    seed=1234,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn decentralized_inventory_control_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    periods: usize,
    demand_mean: f64,
    demand_smoothing_factors: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    seed: u64,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_mean,
        demand_distribution,
        demand_smoothing_factors,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
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
    on_hand_inventory,
    backlog,
    shipment_pipelines,
    order_pipelines,
    last_received_shipments,
    last_received_orders,
    forecast_orders,
    last_actions,
    periods,
    demand_mean,
    demand_smoothing_factors,
    holding_costs,
    backlog_costs,
    seeds,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn decentralized_inventory_control_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    periods: usize,
    demand_mean: f64,
    demand_smoothing_factors: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    seeds: Vec<u64>,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_mean,
        demand_distribution,
        demand_smoothing_factors,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
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
    on_hand_inventory,
    backlog,
    shipment_pipelines,
    order_pipelines,
    last_received_shipments,
    last_received_orders,
    forecast_orders,
    last_actions,
    customer_demands,
    demand_smoothing_factors,
    holding_costs,
    backlog_costs,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn decentralized_inventory_control_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    customer_demands: Vec<usize>,
    demand_smoothing_factors: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    let config = DecentralizedInventoryControlRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: customer_demands.len(),
        customer_demand_model: DemandModel {
            kind: parse_demand_distribution_kind("deterministic")?,
            param1: 0.0,
        },
        demand_smoothing_factors,
        holding_costs,
        backlog_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &customer_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand_inventory,
    backlog,
    shipment_pipelines,
    order_pipelines,
    last_received_shipments,
    last_received_orders,
    forecast_orders,
    last_actions,
    customer_demands,
    demand_smoothing_factors,
    holding_costs,
    backlog_costs,
    discount_factor=0.99
))]
fn decentralized_inventory_control_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    customer_demands: Vec<usize>,
    demand_smoothing_factors: Vec<f64>,
    holding_costs: Vec<f64>,
    backlog_costs: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        &customer_demands,
        &demand_smoothing_factors,
        &holding_costs,
        &backlog_costs,
        discount_factor,
    )
}

#[pyfunction]
fn decentralized_inventory_control_base_stock_orders(
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    realized_customer_demand: usize,
    base_stock_levels: Vec<usize>,
) -> PyResult<Vec<usize>> {
    let state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    let observed_orders =
        crate::problems::decentralized_inventory_control::env::current_received_orders(
            &state,
            realized_customer_demand,
        )?;
    base_stock_orders(&state, &observed_orders, &base_stock_levels)
}

#[pyfunction]
fn decentralized_inventory_control_sterman_anchor_adjust_orders(
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_orders: Vec<f64>,
    last_actions: Vec<usize>,
    realized_customer_demand: usize,
    target_positions: Vec<f64>,
    adjustment_times: Vec<f64>,
    supply_line_weights: Vec<f64>,
) -> PyResult<Vec<usize>> {
    let state = build_initial_state(
        &on_hand_inventory,
        &backlog,
        &shipment_pipelines,
        &order_pipelines,
        &last_received_shipments,
        &last_received_orders,
        &forecast_orders,
        &last_actions,
    )?;
    let observed_orders =
        crate::problems::decentralized_inventory_control::env::current_received_orders(
            &state,
            realized_customer_demand,
        )?;
    sterman_anchor_adjust_orders(
        &state,
        &observed_orders,
        &target_positions,
        &adjustment_times,
        &supply_line_weights,
    )
}

#[pyfunction]
fn decentralized_inventory_control_classic_sterman_literature_summary(
    py: Python<'_>,
) -> PyResult<PyObject> {
    let summary = simulate_classic_sterman_benchmark();
    let dict = PyDict::new_bound(py);
    dict.set_item("per_agent_costs", summary.per_agent_costs.to_vec())?;
    dict.set_item("total_cost", summary.total_cost)?;
    Ok(dict.into())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_base_stock_orders,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_sterman_anchor_adjust_orders,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decentralized_inventory_control_classic_sterman_literature_summary,
        m
    )?)?;
    Ok(())
}
