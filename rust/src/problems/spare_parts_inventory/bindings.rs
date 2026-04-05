use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::spare_parts_inventory::heuristics::{
    base_stock_order_quantity, lead_time_mean_cover_order_quantity,
    lead_time_mean_cover_target, policy_rollout_from_paths, simulate_policy,
};
use crate::problems::spare_parts_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    SparePartsInventoryRolloutConfig,
};

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
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
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
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
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
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    realized_failures,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    realized_failures: Vec<usize>,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_failures.len(),
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
        procurement_cost,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &realized_failures)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    realized_failures,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    discount_factor=0.99
))]
fn spare_parts_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    realized_failures: Vec<usize>,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        installed_base,
        &realized_failures,
        holding_cost,
        downtime_cost,
        procurement_cost,
        failure_probability,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    replications=1000,
    seed=1234,
    discount_factor=0.99
))]
fn spare_parts_inventory_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    replications: usize,
    seed: u64,
    discount_factor: f64,
) -> PyResult<(f64, f64)> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
        procurement_cost,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (on_hand_inventory, backlog, procurement_pipeline, repair_pipeline, installed_base, base_stock_level))]
fn spare_parts_inventory_base_stock_order(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    base_stock_level: usize,
) -> PyResult<usize> {
    let state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    base_stock_order_quantity(&state, base_stock_level)
}

#[pyfunction]
#[pyo3(signature = (
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    failure_probability,
    safety_buffer
))]
fn spare_parts_inventory_lead_time_mean_cover_order(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    failure_probability: f64,
    safety_buffer: f64,
) -> PyResult<usize> {
    let state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    lead_time_mean_cover_order_quantity(
        &state,
        installed_base,
        failure_probability,
        safety_buffer,
    )
}

#[pyfunction]
#[pyo3(signature = (installed_base, failure_probability, procurement_lead_time, safety_buffer))]
fn spare_parts_inventory_lead_time_mean_cover_target(
    installed_base: usize,
    failure_probability: f64,
    procurement_lead_time: usize,
    safety_buffer: f64,
) -> PyResult<usize> {
    lead_time_mean_cover_target(
        installed_base,
        failure_probability,
        procurement_lead_time,
        safety_buffer,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spare_parts_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_base_stock_order, m)?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_lead_time_mean_cover_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_lead_time_mean_cover_target,
        m
    )?)?;
    Ok(())
}
