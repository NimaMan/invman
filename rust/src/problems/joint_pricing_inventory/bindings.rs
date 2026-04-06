use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::joint_pricing_inventory::demand::{
    parse_demand_distribution_kind, validate_price_ladder,
};
use crate::problems::joint_pricing_inventory::env::{build_policy_state, step_state};
use crate::problems::joint_pricing_inventory::heuristics::{
    inventory_sensitive_base_stock_action, policy_rollout_from_demands, simulate_policy,
    static_price_base_stock_action, PolicySimulationSummary,
};
use crate::problems::joint_pricing_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_demands,
    JointPricingInventoryRolloutConfig,
};

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    periods: usize,
    demand_kind: &str,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<JointPricingInventoryRolloutConfig> {
    validate_price_ladder(&price_levels, &demand_means)?;
    Ok(JointPricingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_kind: parse_demand_distribution_kind(demand_kind)?,
        price_levels,
        demand_means,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_order_quantity,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    })
}

fn simulation_summary_to_py(py: Python<'_>, summary: &PolicySimulationSummary) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("mean_discounted_cost", summary.mean_discounted_cost)?;
    dict.set_item("std_discounted_cost", summary.std_discounted_cost)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn joint_pricing_inventory_build_policy_state(
    inventory_level: usize,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    periods: usize,
    max_order_quantity: usize,
) -> PyResult<Vec<f32>> {
    let state = build_initial_state(inventory_level)?;
    build_policy_state(
        &state,
        &price_levels,
        &demand_means,
        periods,
        max_order_quantity,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    order_quantity,
    price_index,
    realized_demand,
    price_levels,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit
))]
fn joint_pricing_inventory_step(
    py: Python<'_>,
    inventory_level: usize,
    order_quantity: usize,
    price_index: usize,
    realized_demand: usize,
    price_levels: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
) -> PyResult<PyObject> {
    let state = build_initial_state(inventory_level)?;
    let outcome = step_state(
        &state,
        order_quantity,
        price_index,
        realized_demand,
        &price_levels,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("next_period", outcome.next_state.period)?;
    dict.set_item("next_inventory_level", outcome.next_state.inventory_level)?;
    dict.set_item("selling_price", outcome.selling_price)?;
    dict.set_item("sales", outcome.sales)?;
    dict.set_item("lost_sales", outcome.lost_sales)?;
    dict.set_item("revenue", outcome.revenue)?;
    dict.set_item("procurement_cost", outcome.procurement_cost)?;
    dict.set_item("holding_cost", outcome.holding_cost)?;
    dict.set_item("stockout_cost", outcome.stockout_cost)?;
    dict.set_item("period_cost", outcome.period_cost)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn joint_pricing_inventory_static_price_base_stock_action(
    inventory_level: usize,
    order_up_to: usize,
    price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    static_price_base_stock_action(
        inventory_level,
        order_up_to,
        price_index,
        max_order_quantity,
        num_prices,
    )
}

#[pyfunction]
fn joint_pricing_inventory_inventory_sensitive_base_stock_action(
    inventory_level: usize,
    order_up_to: usize,
    markdown_threshold: usize,
    high_price_index: usize,
    low_price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    inventory_sensitive_base_stock_action(
        inventory_level,
        order_up_to,
        markdown_threshold,
        high_price_index,
        low_price_index,
        max_order_quantity,
        num_prices,
    )
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_level,
    periods,
    demand_kind,
    price_levels,
    demand_means,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_order_quantity,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_pricing_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    periods: usize,
    demand_kind: &str,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_order_quantity: usize,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(inventory_level)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        price_levels,
        demand_means,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_order_quantity,
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
    inventory_level,
    periods,
    demand_kind,
    price_levels,
    demand_means,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_order_quantity,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_pricing_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    periods: usize,
    demand_kind: &str,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_order_quantity: usize,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(inventory_level)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        price_levels,
        demand_means,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_order_quantity,
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
    inventory_level,
    realized_demands,
    price_levels,
    demand_means,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_order_quantity,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_pricing_inventory_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    realized_demands: Vec<usize>,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(inventory_level)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        realized_demands.len(),
        "deterministic",
        price_levels,
        demand_means,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_order_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    rollout_from_demands(&flat_params, &config, &initial_state, &realized_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    realized_demands,
    price_levels,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    max_order_quantity,
    discount_factor=0.99,
    salvage_value_per_unit=0.0
))]
fn joint_pricing_inventory_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<f64>,
    inventory_level: usize,
    realized_demands: Vec<usize>,
    price_levels: Vec<f64>,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(inventory_level)?;
    policy_rollout_from_demands(
        policy_name,
        &params,
        &initial_state,
        &realized_demands,
        &price_levels,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        max_order_quantity,
        discount_factor,
        salvage_value_per_unit,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    periods,
    replications,
    seed,
    price_levels,
    demand_means,
    demand_kind,
    procurement_cost_per_unit,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    max_order_quantity,
    discount_factor=0.99,
    salvage_value_per_unit=0.0
))]
fn joint_pricing_inventory_simulate_policy(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<f64>,
    inventory_level: usize,
    periods: usize,
    replications: usize,
    seed: u64,
    price_levels: Vec<f64>,
    demand_means: Vec<f64>,
    demand_kind: &str,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<PyObject> {
    let initial_state = build_initial_state(inventory_level)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        &price_levels,
        &demand_means,
        parse_demand_distribution_kind(demand_kind)?,
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        max_order_quantity,
        discount_factor,
        salvage_value_per_unit,
    )?;
    simulation_summary_to_py(py, &summary)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(joint_pricing_inventory_build_policy_state, m)?)?;
    m.add_function(wrap_pyfunction!(joint_pricing_inventory_step, m)?)?;
    m.add_function(wrap_pyfunction!(
        joint_pricing_inventory_static_price_base_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_pricing_inventory_inventory_sensitive_base_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(joint_pricing_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        joint_pricing_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_pricing_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_pricing_inventory_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(joint_pricing_inventory_simulate_policy, m)?)?;
    Ok(())
}
