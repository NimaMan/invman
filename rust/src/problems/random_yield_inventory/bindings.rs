use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::random_yield_inventory::demand::parse_demand_distribution_kind;
use crate::problems::random_yield_inventory::heuristics::{
    policy_rollout_from_paths, simulate_policy, weighted_newsvendor_order_quantity,
    yield_inflated_base_stock_order_quantity, yield_inflated_base_stock_parameters,
};
use crate::problems::random_yield_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    RandomYieldInventoryRolloutConfig,
};

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    seed=1234,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    seed: u64,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_mean,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
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
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    seeds,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    seeds: Vec<u64>,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_mean,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
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
    initial_inventory_level,
    pipeline_orders,
    demands,
    arrival_outcomes,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    arrival_outcomes: Vec<bool>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: demands.len(),
        demand_mean,
        demand_kind: crate::problems::random_yield_inventory::demand::DemandDistributionKind::Deterministic,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &demands, &arrival_outcomes)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_level,
    pipeline_orders,
    demands,
    arrival_outcomes,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    discount_factor=0.99
))]
fn random_yield_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    arrival_outcomes: Vec<bool>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        demand_mean,
        &demands,
        &arrival_outcomes,
        holding_cost,
        shortage_cost,
        procurement_cost,
        success_probability,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    replications=1000,
    seed=1234,
    discount_factor=0.99,
    demand_distribution="poisson"
))]
fn random_yield_inventory_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    replications: usize,
    seed: u64,
    discount_factor: f64,
    demand_distribution: &str,
) -> PyResult<(f64, f64)> {
    let summary = simulate_policy(
        policy_name,
        &params,
        initial_inventory_level,
        &pipeline_orders,
        periods,
        replications,
        seed,
        demand_mean,
        parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (
    demand_mean,
    success_probability,
    lead_time,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_linear_inflation_parameters(
    demand_mean: f64,
    success_probability: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<(f64, f64)> {
    yield_inflated_base_stock_parameters(
        demand_mean,
        success_probability,
        lead_time,
        holding_cost,
        shortage_cost,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    pipeline_orders,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_weighted_newsvendor_order(
    inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    let state = build_initial_state(inventory_level, &pipeline_orders)?;
    weighted_newsvendor_order_quantity(
        &state,
        demand_mean,
        success_probability,
        holding_cost,
        shortage_cost,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    pipeline_orders,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_yield_inflated_base_stock_order(
    inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    let state = build_initial_state(inventory_level, &pipeline_orders)?;
    yield_inflated_base_stock_order_quantity(
        &state,
        demand_mean,
        success_probability,
        holding_cost,
        shortage_cost,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(random_yield_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(random_yield_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_linear_inflation_parameters,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_weighted_newsvendor_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_yield_inflated_base_stock_order,
        m
    )?)?;
    Ok(())
}
