use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::procurement_removal_inventory::demand::{
    parse_demand_distribution_kind,
};
use crate::problems::procurement_removal_inventory::env::{
    build_raw_state, initialize_state, step_state,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, policy_rollout, policy_rollout_from_demands,
    returnability_buffer_interval_stock_action, simulate_policy, PolicySimulationSummary,
};
use crate::problems::procurement_removal_inventory::rollout::{
    population_rollout, rollout, rollout_from_demands, ProcurementRemovalRolloutConfig,
};

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<ProcurementRemovalRolloutConfig> {
    Ok(ProcurementRemovalRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_kind: parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
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
    dict.set_item("min_discounted_cost", summary.min_discounted_cost)?;
    dict.set_item("max_discounted_cost", summary.max_discounted_cost)?;
    dict.set_item("num_seeds", summary.num_seeds)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn procurement_removal_inventory_build_raw_state(
    inventory_level: usize,
    returnable_inventory: usize,
) -> PyResult<Vec<f32>> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    build_raw_state(&state)
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    returnable_inventory,
    purchase_quantity,
    removal_quantity,
    realized_demand,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit
))]
fn procurement_removal_inventory_step(
    py: Python<'_>,
    inventory_level: usize,
    returnable_inventory: usize,
    purchase_quantity: usize,
    removal_quantity: usize,
    realized_demand: usize,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
) -> PyResult<PyObject> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    let outcome = step_state(
        &state,
        purchase_quantity,
        removal_quantity,
        realized_demand,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("next_period", outcome.next_state.period)?;
    dict.set_item("next_inventory_level", outcome.next_state.inventory_level)?;
    dict.set_item(
        "next_returnable_inventory",
        outcome.next_state.returnable_inventory,
    )?;
    dict.set_item("returned_units", outcome.returned_units)?;
    dict.set_item("liquidated_units", outcome.liquidated_units)?;
    dict.set_item("sales", outcome.sales)?;
    dict.set_item("shortage", outcome.shortage)?;
    dict.set_item("purchase_cost", outcome.purchase_cost)?;
    dict.set_item("removal_credit", outcome.removal_credit)?;
    dict.set_item("holding_cost", outcome.holding_cost)?;
    dict.set_item("shortage_cost", outcome.shortage_cost)?;
    dict.set_item("period_cost", outcome.period_cost)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn procurement_removal_inventory_interval_stock_action(
    inventory_level: usize,
    returnable_inventory: usize,
    order_up_to: usize,
    remove_down_to: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    interval_stock_action(
        &state,
        order_up_to,
        remove_down_to,
        max_purchase_quantity,
        max_removal_quantity,
    )
}

#[pyfunction]
fn procurement_removal_inventory_returnability_buffer_interval_stock_action(
    inventory_level: usize,
    returnable_inventory: usize,
    order_up_to: usize,
    remove_down_to: usize,
    returnable_buffer: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    returnability_buffer_interval_stock_action(
        &state,
        order_up_to,
        remove_down_to,
        returnable_buffer,
        max_purchase_quantity,
        max_removal_quantity,
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
    returnable_inventory,
    periods,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
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
    returnable_inventory,
    periods,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
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
    returnable_inventory,
    demands,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    demands: Vec<usize>,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        demands.len(),
        "deterministic",
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    rollout_from_demands(&flat_params, &config, &initial_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    periods,
    seed,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_policy_rollout(
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    seed: u64,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    policy_rollout(
        policy_name,
        &params,
        &initial_state,
        periods,
        seed,
        parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    periods,
    seeds,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_simulate_policy(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    seeds: Vec<u64>,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<PyObject> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        &seeds,
        parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )?;
    simulation_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    demands,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    demands: Vec<usize>,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    policy_rollout_from_demands(
        policy_name,
        &params,
        &initial_state,
        &demands,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_build_raw_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(procurement_removal_inventory_step, m)?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_interval_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_returnability_buffer_interval_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_policy_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_simulate_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_policy_rollout_from_demands,
        m
    )?)?;
    Ok(())
}
